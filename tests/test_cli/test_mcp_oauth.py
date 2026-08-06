from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

import pytest

from mcp.shared.auth import OAuthClientInformationFull, OAuthToken

from ness_agent.mcp import MCPManager
from ness_cli import mcp_oauth
from ness_cli.mcp_oauth import (
    LoopbackOAuthCallback,
    MCPOAuthService,
    ProjectOAuthTokenStorage,
    PinnedScopeOAuthClientProvider,
    credential_id,
    parse_oauth_callback_url,
)


def _write_config(tmp_path: Path, spec: dict) -> MCPManager:
    path = tmp_path / "mcp.json"
    path.write_text(json.dumps({"mcpServers": {"remote": spec}}), encoding="utf-8")
    manager = MCPManager(path, project_root=tmp_path)
    manager.load()
    return manager


def test_cursor_and_claude_oauth_normalize_and_redact(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("CLIENT_SECRET", "secret-from-environment")
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "cursor": {
                        "url": "https://example.com/mcp",
                        "auth": {
                            "CLIENT_ID": "cursor-id",
                            "CLIENT_SECRET": "${env:CLIENT_SECRET}",
                            "scopes": ["read", "write"],
                        },
                    },
                    "claude": {
                        "type": "http",
                        "url": "https://example.net/mcp",
                        "oauth": {
                            "clientId": "claude-id",
                            "callbackPort": 9191,
                            "scopes": "files:read files:write",
                            "tokenEndpointAuthMethod": "client_secret_basic",
                        },
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    manager = MCPManager(path, project_root=tmp_path)
    preview = manager.load()
    cursor = manager.server_spec("cursor")
    claude = manager.server_spec("claude")
    assert cursor and cursor.oauth
    assert cursor.oauth.callback_port == 8787
    assert cursor.oauth.scopes == ("read", "write")
    assert cursor.oauth.client_secret == "secret-from-environment"
    assert claude and claude.oauth
    assert claude.oauth.callback_port == 9191
    assert claude.oauth.scopes == ("files:read", "files:write")
    assert "secret-from-environment" not in "\n".join(preview.servers)


def test_invalid_oauth_combinations_are_visible(tmp_path: Path):
    path = tmp_path / "mcp.json"
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "both": {
                        "url": "https://example.com/mcp",
                        "auth": {"CLIENT_ID": "x"},
                        "oauth": {},
                    },
                    "header": {
                        "url": "https://example.com/mcp",
                        "headers": {"Authorization": "Bearer x"},
                        "oauth": {},
                    },
                    "metadata": {
                        "url": "https://example.com/mcp",
                        "oauth": {"authServerMetadataUrl": "https://auth.example.com"},
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    manager = MCPManager(path, project_root=tmp_path)
    manager.load()
    assert all(value["status"] == "error" for value in manager.servers.values())
    assert "both auth and oauth" in manager.servers["both"]["error"]
    assert "Authorization header" in manager.servers["header"]["error"]
    assert "not supported" in manager.servers["metadata"]["error"]


def test_missing_oauth_tokens_marks_auth_required_without_network(tmp_path: Path):
    manager = _write_config(
        tmp_path,
        {"url": "https://example.com/mcp", "oauth": {"clientId": "client"}},
    )
    service = MCPOAuthService(project_root=tmp_path, config_dir=tmp_path / "config")
    manager.http_auth_factory = service.startup_auth

    async def exercise():
        await manager.start()
        assert manager.servers["remote"]["status"] == "auth_required"
        assert "ness mcp login remote" in manager.servers["remote"]["error"]
        await manager.stop()

    asyncio.run(exercise())


def test_fallback_storage_is_atomic_0600_and_round_trips(tmp_path: Path, monkeypatch):
    async def unavailable(identity):
        return False, None

    monkeypatch.setattr(mcp_oauth, "_keyring_get", unavailable)
    storage = ProjectOAuthTokenStorage(
        config_dir=tmp_path,
        identity="identity",
    )
    token = OAuthToken(access_token="access-secret", refresh_token="refresh-secret", expires_in=60)
    client = OAuthClientInformationFull(
        client_id="dynamic-id",
        client_secret="dynamic-secret",
        redirect_uris=["http://localhost:8765/callback"],
    )

    async def exercise():
        await storage.set_tokens(token)
        await storage.set_client_info(client)
        assert (await storage.get_tokens()).access_token == "access-secret"
        assert storage.token_expiry_time is not None
        await storage.set_tokens(OAuthToken(access_token="refreshed", expires_in=60))
        refreshed = await storage.get_tokens()
        assert refreshed.access_token == "refreshed"
        assert refreshed.refresh_token == "refresh-secret"
        assert (await storage.get_client_info()).client_id == "dynamic-id"
        assert await storage.has_credentials()
        await storage.clear()
        assert not await storage.has_credentials()

    asyncio.run(exercise())
    assert storage.fallback_path.exists()
    assert os.stat(storage.fallback_path).st_mode & 0o777 == 0o600
    assert "No usable system keyring" in storage.warnings[0]


def test_static_client_info_is_not_persisted_to_fallback(tmp_path: Path, monkeypatch):
    async def unavailable(identity):
        return False, None

    monkeypatch.setattr(mcp_oauth, "_keyring_get", unavailable)
    static = OAuthClientInformationFull(
        client_id="static-id",
        client_secret="do-not-duplicate",
        redirect_uris=["http://localhost:8787/callback"],
    )
    storage = ProjectOAuthTokenStorage(
        config_dir=tmp_path,
        identity="static",
        static_client_info=static,
    )

    async def exercise():
        await storage.set_client_info(static)
        await storage.set_tokens(OAuthToken(access_token="token"))
        assert (await storage.get_client_info()).client_id == "static-id"

    asyncio.run(exercise())
    text = storage.fallback_path.read_text(encoding="utf-8")
    assert "do-not-duplicate" not in text


def test_corrupt_fallback_fails_closed_without_overwrite(tmp_path: Path, monkeypatch):
    async def unavailable(identity):
        return False, None

    monkeypatch.setattr(mcp_oauth, "_keyring_get", unavailable)
    storage = ProjectOAuthTokenStorage(config_dir=tmp_path, identity="corrupt")
    storage.fallback_path.write_text("not-json", encoding="utf-8")
    before = storage.fallback_path.read_bytes()

    async def exercise():
        assert await storage.get_tokens() is None
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            await storage.set_tokens(OAuthToken(access_token="new"))

    asyncio.run(exercise())
    assert storage.fallback_path.read_bytes() == before


def test_credential_identity_is_project_scoped(tmp_path: Path):
    manager = _write_config(tmp_path, {"url": "https://example.com/mcp"})
    spec = manager.server_spec("remote")
    assert spec
    assert credential_id(tmp_path, spec) != credential_id(tmp_path / "other", spec)


def test_pinned_scope_adapter_reapplies_configured_scope(tmp_path: Path, monkeypatch):
    from mcp.client.auth import OAuthClientProvider
    from mcp.shared.auth import OAuthClientMetadata

    async def unavailable(identity):
        return False, None

    async def fake_authorization(self):
        return self.context.client_metadata.scope

    monkeypatch.setattr(mcp_oauth, "_keyring_get", unavailable)
    monkeypatch.setattr(OAuthClientProvider, "_perform_authorization", fake_authorization)
    storage = ProjectOAuthTokenStorage(config_dir=tmp_path, identity="scope")
    provider = PinnedScopeOAuthClientProvider(
        server_url="https://example.com/mcp",
        client_metadata=OAuthClientMetadata(
            redirect_uris=["http://localhost:8765/callback"],
            scope="discovered",
        ),
        storage=storage,
        pinned_scope="read write",
    )
    assert asyncio.run(provider._perform_authorization()) == "read write"


def test_parse_callback_and_loopback_receiver():
    assert parse_oauth_callback_url("http://localhost:1/callback?code=abc&state=xyz") == (
        "abc",
        "xyz",
    )

    async def exercise():
        callback = LoopbackOAuthCallback(timeout=2)
        await callback.start()
        parts = urlsplit(callback.redirect_uri)
        reader, writer = await asyncio.open_connection("127.0.0.1", parts.port)
        writer.write(
            b"GET /callback?code=hello&state=state HTTP/1.1\r\nHost: localhost\r\n\r\n"
        )
        await writer.drain()
        response = await reader.read()
        writer.close()
        await writer.wait_closed()
        assert b"200 OK" in response
        assert await callback.callback_handler() == ("hello", "state")
        await callback.close()

    from urllib.parse import urlsplit

    asyncio.run(exercise())


def test_stored_bearer_token_authenticates_real_streamable_http(
    tmp_path: Path, monkeypatch
):
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import PlainTextResponse

    async def unavailable(identity):
        return False, None

    monkeypatch.setattr(mcp_oauth, "_keyring_get", unavailable)
    server_mcp = FastMCP("oauth-http", stateless_http=True, json_response=True)

    @server_mcp.tool()
    def secured_echo(message: str) -> str:
        return message

    async def exercise():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        app = server_mcp.streamable_http_app()

        class RequireBearer(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.headers.get("Authorization") != "Bearer valid-access":
                    return PlainTextResponse("unauthorized", status_code=401)
                return await call_next(request)

        app.add_middleware(RequireBearer)
        uvicorn_server = uvicorn.Server(
            uvicorn.Config(app, log_level="error", lifespan="on")
        )
        server_task = asyncio.create_task(uvicorn_server.serve(sockets=[sock]))
        for _ in range(100):
            if uvicorn_server.started:
                break
            await asyncio.sleep(0.01)

        manager = _write_config(
            tmp_path,
            {
                "url": f"http://127.0.0.1:{port}/mcp",
                "oauth": {"clientId": "client"},
            },
        )
        service = MCPOAuthService(
            project_root=tmp_path,
            config_dir=tmp_path / "config",
        )
        spec = manager.server_spec("remote")
        assert spec
        await service.storage_for(spec).set_tokens(
            OAuthToken(access_token="valid-access", expires_in=3600)
        )
        manager.http_auth_factory = service.startup_auth
        try:
            await manager.start()
            assert manager.servers["remote"]["status"] == "connected"
            result = await manager.call(
                "remote", "secured_echo", {"message": "authenticated"}
            )
            assert result.startswith("authenticated")
        finally:
            await manager.stop()
            uvicorn_server.should_exit = True
            await server_task

    asyncio.run(exercise())
