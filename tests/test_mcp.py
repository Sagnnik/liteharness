from __future__ import annotations

import asyncio
import json
import os
import socket
from pathlib import Path

import pytest

from ness_agent.mcp import MCPManager, _sanitize_tool_args


def _write(path: Path, data) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _manager(tmp_path: Path, data=None) -> MCPManager:
    path = tmp_path / "mcp.json"
    if data is not None:
        _write(path, data)
    return MCPManager(path, project_root=tmp_path)


def test_load_missing_and_empty_are_not_errors(tmp_path: Path):
    missing = _manager(tmp_path)
    assert not missing.load().has_runnable_servers
    assert missing.startup_summary() == ("MCP: none configured", "none")

    empty = _manager(tmp_path, {"mcpServers": {}})
    assert not empty.load().has_runnable_servers
    assert empty.startup_summary() == ("MCP: none configured", "none")


@pytest.mark.parametrize("content", ["[", "[]"])
def test_load_reports_top_level_errors(tmp_path: Path, content: str):
    path = tmp_path / "mcp.json"
    path.write_text(content, encoding="utf-8")
    manager = MCPManager(path, project_root=tmp_path)
    manager.load()
    message, level = manager.startup_summary()
    assert level == "warn"
    assert "config error" in message


def test_legacy_servers_key_is_rejected(tmp_path: Path):
    manager = _manager(tmp_path, {"servers": {"echo": {"command": "python"}}})
    preview = manager.load()
    assert not preview.has_runnable_servers
    assert "'servers' is not supported" in manager.status()


def test_invalid_server_is_isolated(tmp_path: Path):
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "ok": {"command": "python", "args": ["server.py"]},
                "bad": "not an object",
            },
        },
    )
    preview = manager.load()
    assert preview.has_runnable_servers
    assert set(manager.servers) == {"ok", "bad"}
    assert manager.servers["bad"]["status"] == "error"


def test_load_does_not_start_servers(tmp_path: Path, monkeypatch):
    manager = _manager(tmp_path, {"mcpServers": {"echo": {"command": "python"}}})

    async def forbidden(*args, **kwargs):
        raise AssertionError("load attempted to connect")

    monkeypatch.setattr(manager, "start_server", forbidden)
    assert manager.load().has_runnable_servers


def test_stdio_normalization_and_interpolation(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MCP_TOKEN", "secret-token")
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "stdio": {
                    "type": "stdio",
                    "command": ["python", "-m"],
                    "args": [
                        "server",
                        "${workspaceFolderBasename}",
                        "${MISSING:-fallback}",
                        "${MCP_TOKEN}",
                    ],
                    "cwd": "${workspaceFolder}/subdir",
                    "env": {"TOKEN": "${env:MCP_TOKEN}", "PATH_SEP": "${/}"},
                }
            }
        },
    )
    preview = manager.load()
    spec = manager._specs["stdio"]
    assert spec.command == "python"
    assert spec.args == ("-m", "server", tmp_path.name, "fallback", "secret-token")
    assert spec.cwd == (tmp_path / "subdir").resolve()
    assert dict(spec.env)["TOKEN"] == "secret-token"
    assert dict(spec.env)["PATH_SEP"] == os.sep
    assert "secret-token" not in "\n".join(preview.servers)


def test_interpolation_is_non_recursive_and_missing_variable_isolated(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OUTER", "${INNER}")
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "nonrecursive": {"command": "${OUTER}"},
                "missing": {"command": "${DOES_NOT_EXIST}"},
            }
        },
    )
    manager.load()
    assert manager._specs["nonrecursive"].command == "${INNER}"
    assert manager.servers["missing"]["status"] == "error"
    assert "DOES_NOT_EXIST" in manager.servers["missing"]["error"]


def test_env_file_precedence_and_no_arbitrary_env_inheritance(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_ARBITRARY_PARENT_VALUE", "must-not-leak")
    (tmp_path / ".env").write_text("FROM_FILE=yes\nOVERRIDE=file\n", encoding="utf-8")
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "stdio": {
                    "command": "python",
                    "envFile": ".env",
                    "env": {"OVERRIDE": "explicit"},
                }
            }
        },
    )
    manager.load()
    env = dict(manager._specs["stdio"].env)
    assert env["FROM_FILE"] == "yes"
    assert env["OVERRIDE"] == "explicit"
    assert "NESS_ARBITRARY_PARENT_VALUE" not in env


def test_env_file_is_stdio_only_and_must_exist(tmp_path: Path):
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "remote": {"url": "https://example.com/mcp", "envFile": ".env"},
                "stdio": {"command": "python", "envFile": "missing.env"},
            }
        },
    )
    manager.load()
    assert manager.servers["remote"]["status"] == "error"
    assert manager.servers["stdio"]["status"] == "error"


@pytest.mark.parametrize(
    ("spec", "error"),
    [
        ({"command": "x", "url": "https://example.com"}, "both command and url"),
        ({"type": "sse", "url": "https://example.com/sse"}, "unsupported transport"),
        ({"type": "http", "command": "x"}, "requires url"),
        ({"url": "ftp://example.com"}, "http(s) URL"),
        ({"url": "http://example.com:not-a-port/mcp"}, "malformed"),
        ({"url": "https://user:secret@example.com/mcp"}, "embedded credentials"),
    ],
)
def test_invalid_transport_shapes(tmp_path: Path, spec: dict, error: str):
    manager = _manager(tmp_path, {"mcpServers": {"bad": spec}})
    manager.load()
    assert error in manager.servers["bad"]["error"]


def test_http_cursor_and_claude_shapes(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("TOKEN", "top-secret")
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "cursor": {
                    "url": "https://example.com/mcp?workspace=1",
                    "headers": {"Authorization": "Bearer ${TOKEN}"},
                },
                "claude": {"type": "http", "url": "http://localhost:8765/mcp"},
            }
        },
    )
    preview = manager.load()
    assert manager._specs["cursor"].transport == "http"
    assert dict(manager._specs["cursor"].headers)["Authorization"] == "Bearer top-secret"
    summary = "\n".join(preview.servers)
    assert "top-secret" not in summary
    assert "workspace=1" not in summary


def test_fingerprint_is_canonical_and_tracks_effective_target(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("MCP_COMMAND", "python")
    path = tmp_path / "mcp.json"
    first = {"mcpServers": {"one": {"command": "${MCP_COMMAND}", "env": {"TOKEN": "${TOKEN:-x}"}}}}
    _write(path, first)
    fp1 = MCPManager(path, project_root=tmp_path).load().fingerprint
    path.write_text('{\n  "mcpServers": {"one": {"env": {"TOKEN": "${TOKEN:-x}"}, "command": "${MCP_COMMAND}"}}\n}', encoding="utf-8")
    fp2 = MCPManager(path, project_root=tmp_path).load().fingerprint
    assert fp1 == fp2
    monkeypatch.setenv("MCP_COMMAND", "uv")
    fp3 = MCPManager(path, project_root=tmp_path).load().fingerprint
    assert fp3 != fp1


def test_mark_untrusted_and_stop_reset_lifecycle(tmp_path: Path):
    manager = _manager(tmp_path, {"mcpServers": {"one": {"command": "python"}}})
    manager.mark_untrusted()
    assert manager.servers["one"]["status"] == "pending_trust"
    asyncio.run(manager.stop())
    assert not manager.servers
    assert manager.load().has_runnable_servers


def test_stdio_echo_integration(tmp_path: Path):
    server = Path(__file__).parent / "mcp_echo_server.py"
    manager = _manager(
        tmp_path,
        {"mcpServers": {"echo": {"command": "python", "args": [str(server)]}}},
    )

    async def exercise():
        try:
            await manager.start()
            assert manager.servers["echo"]["status"] == "connected"
            assert "mcp__echo__echo" in manager.tools
            assert (await manager.call("echo", "echo", {"message": "hello"})).startswith("hello")
        finally:
            await manager.stop()

    asyncio.run(exercise())


def test_streamable_http_integration(tmp_path: Path):
    import uvicorn
    from mcp.server.fastmcp import FastMCP
    from starlette.middleware.base import BaseHTTPMiddleware
    from starlette.responses import PlainTextResponse

    server_mcp = FastMCP("http-echo", stateless_http=True, json_response=True)

    @server_mcp.tool()
    def echo_http(message: str) -> str:
        return message

    async def exercise():
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", 0))
        sock.listen(128)
        port = sock.getsockname()[1]
        app = server_mcp.streamable_http_app()

        class RequireTestHeader(BaseHTTPMiddleware):
            async def dispatch(self, request, call_next):
                if request.headers.get("X-Test") != "configured":
                    return PlainTextResponse("missing test header", status_code=401)
                return await call_next(request)

        app.add_middleware(RequireTestHeader)
        uvicorn_server = uvicorn.Server(
            uvicorn.Config(
                app,
                log_level="error",
                lifespan="on",
            )
        )
        server_task = asyncio.create_task(uvicorn_server.serve(sockets=[sock]))
        for _ in range(100):
            if uvicorn_server.started:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("HTTP test server did not start")

        manager = _manager(
            tmp_path,
            {
                "mcpServers": {
                    "remote": {
                        "type": "http",
                        "url": f"http://127.0.0.1:{port}/mcp",
                        "headers": {"X-Test": "configured"},
                    }
                }
            },
        )
        try:
            await manager.start()
            assert manager.servers["remote"]["status"] == "connected"
            assert "mcp__remote__echo_http" in manager.tools
            result = await manager.call(
                "remote", "echo_http", {"message": "hello-http"}
            )
            assert result.startswith("hello-http")
        finally:
            await manager.stop()
            uvicorn_server.should_exit = True
            await server_task

    asyncio.run(exercise())


def test_connection_errors_are_redacted_and_do_not_block_other_servers(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("REMOTE_TOKEN", "super-secret-value")
    manager = _manager(
        tmp_path,
        {
            "mcpServers": {
                "bad": {
                    "url": "https://example.com/mcp",
                    "headers": {"Authorization": "Bearer ${REMOTE_TOKEN}"},
                },
                "good": {"command": "python"},
            }
        },
    )

    async def fake_http(name, spec, stack):
        raise RuntimeError("request failed with Bearer super-secret-value")

    async def fake_stdio(name, spec, stack):
        manager.servers[name] = {
            "status": "connected",
            "description": "",
            "transport": "stdio",
            "tools": [],
        }

    monkeypatch.setattr(manager, "_connect_http", fake_http)
    monkeypatch.setattr(manager, "_connect_stdio", fake_stdio)

    async def exercise():
        try:
            await manager.start()
            assert manager.servers["good"]["status"] == "connected"
            assert manager.servers["bad"]["status"] == "error"
            assert "super-secret-value" not in manager.servers["bad"]["error"]
            assert "[redacted]" in manager.servers["bad"]["error"]
        finally:
            await manager.stop()

    asyncio.run(exercise())


def test_sanitize_tool_args_strips_null_padding_and_empty_nested_objects():
    assert _sanitize_tool_args(
        {
            "limit": 50,
            "cursor": None,
            "query": None,
            "filter": {"createdAt": {"gte": None, "lte": None}},
        }
    ) == {"limit": 50}


def test_sanitize_tool_args_preserves_falsy_non_null_values():
    assert _sanitize_tool_args({"count": 0, "flag": False, "text": "", "items": []}) == {
        "count": 0,
        "flag": False,
        "text": "",
        "items": [],
    }


def test_sanitize_tool_args_leaves_explicit_values_intact():
    assert _sanitize_tool_args(
        {
            "cursor": "abc123",
            "filter": {"createdAt": {"gte": "2026-01-01T00:00:00Z"}},
        }
    ) == {
        "cursor": "abc123",
        "filter": {"createdAt": {"gte": "2026-01-01T00:00:00Z"}},
    }


def test_call_forwards_sanitized_arguments(tmp_path: Path):
    from types import SimpleNamespace
    from typing import Any

    manager = _manager(tmp_path, {"mcpServers": {"echo": {"command": "python"}}})
    captured: dict[str, Any] = {}

    class FakeSession:
        async def call_tool(self, tool_name: str, args: dict[str, Any]):
            captured["tool_name"] = tool_name
            captured["args"] = args
            return SimpleNamespace(isError=False, content=[], structuredContent=None)

    manager.sessions["echo"] = FakeSession()

    async def exercise():
        await manager.call(
            "echo",
            "list_teams",
            {
                "limit": 50,
                "cursor": None,
                "filter": {"createdAt": {"gte": None}},
            },
        )

    asyncio.run(exercise())
    assert captured["tool_name"] == "list_teams"
    assert captured["args"] == {"limit": 50}
