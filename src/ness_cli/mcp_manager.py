"""Ness project configuration and presentation adapter for the MCP SDK runtime."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from dotenv.parser import parse_stream

from ness_agent.mcp import MCPRuntime, MCPServerSpec, validate_mcp_http_url
from ness_cli.terminal import terminal_safe_text

DEFAULT_STARTUP_TIMEOUT = 20.0
_PLACEHOLDER_RE = re.compile(r"\$\{([^{}]+)\}")


@dataclass(frozen=True)
class MCPOAuthSpec:
    source: Literal["cursor", "claude"]
    client_id: str | None = None
    client_secret: str | None = None
    scopes: tuple[str, ...] = ()
    callback_port: int | None = None
    token_endpoint_auth_method: Literal[
        "none", "client_secret_post", "client_secret_basic"
    ] = "none"


@dataclass(frozen=True)
class ProjectMCPServer:
    """CLI metadata wrapped around one resolved SDK connection spec."""

    connection: MCPServerSpec
    oauth: MCPOAuthSpec | None = None
    env_file: Path | None = None

    @property
    def name(self) -> str:
        return self.connection.name

    @property
    def transport(self) -> Literal["stdio", "http"]:
        return self.connection.transport

    @property
    def description(self) -> str:
        return self.connection.description

    @property
    def startup_timeout(self) -> float:
        return self.connection.startup_timeout

    @property
    def command(self) -> str | None:
        return self.connection.command

    @property
    def args(self) -> tuple[str, ...]:
        return self.connection.args

    @property
    def cwd(self) -> Path | None:
        return self.connection.cwd

    @property
    def env(self) -> tuple[tuple[str, str], ...]:
        return self.connection.env

    @property
    def url(self) -> str | None:
        return self.connection.url

    @property
    def headers(self) -> tuple[tuple[str, str], ...]:
        return self.connection.headers

    @property
    def redactions(self) -> tuple[str, ...]:
        return self.connection.redactions

    def with_startup_timeout(self, timeout: float) -> "ProjectMCPServer":
        return replace(
            self,
            connection=replace(self.connection, startup_timeout=timeout),
        )


@dataclass(frozen=True)
class MCPTrustPreview:
    config_path: Path | None
    fingerprint: str | None
    servers: tuple[str, ...]

    @property
    def has_runnable_servers(self) -> bool:
        return bool(self.servers and self.fingerprint)


class MCPConfigError(ValueError):
    pass


ProjectHTTPAuthFactory = Callable[[ProjectMCPServer], Awaitable[Any | None]]


class ProjectMCPManager:
    """Load Ness project MCP config and coordinate an SDK ``MCPRuntime``."""

    def __init__(
        self,
        mcp_file: Path | None = None,
        *,
        project_root: Path = Path.cwd(),
        http_auth_factory: ProjectHTTPAuthFactory | None = None,
        runtime: MCPRuntime | None = None,
    ) -> None:
        self.mcp_file = Path(mcp_file) if mcp_file is not None else None
        self.project_root = Path(project_root).resolve()
        self.http_auth_factory = http_auth_factory
        self.runtime = runtime or MCPRuntime(http_auth_factory=self._runtime_auth)
        if runtime is not None:
            self.runtime.http_auth_factory = self._runtime_auth
        self._servers: dict[str, dict[str, Any]] = {}
        self._specs: dict[str, ProjectMCPServer] = {}
        self._fingerprint_material: dict[str, dict[str, Any]] = {}
        self._config_errors: list[str] = []
        self._loaded = False

    @property
    def servers(self) -> dict[str, dict[str, Any]]:
        self._sync_runtime_states()
        return self._servers

    @property
    def sessions(self) -> dict[str, Any]:
        return self.runtime.sessions

    @property
    def tools(self):  # type: ignore[no-untyped-def]
        return self.runtime.tools

    def load(self) -> MCPTrustPreview:
        """Parse, normalize, and fingerprint project config without connecting."""
        if self._loaded:
            return self.trust_preview
        self._loaded = True
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self._servers.clear()
        if self.mcp_file is None or not self.mcp_file.exists():
            return self.trust_preview
        try:
            config = json.loads(self.mcp_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._config_errors.append(
                f"invalid JSON at line {exc.lineno}, column {exc.colno}"
            )
            return self.trust_preview
        except OSError as exc:
            self._config_errors.append(f"cannot read config: {exc}")
            return self.trust_preview
        if not isinstance(config, dict):
            self._config_errors.append("config root must be a JSON object")
            return self.trust_preview
        if "servers" in config:
            self._config_errors.append("'servers' is not supported; use 'mcpServers'")
        raw_servers = config.get("mcpServers", {})
        if not isinstance(raw_servers, dict):
            self._config_errors.append("'mcpServers' must be a JSON object")
            return self.trust_preview

        for raw_name, raw_spec in raw_servers.items():
            name = str(raw_name)
            if not isinstance(raw_name, str) or not raw_name.strip():
                self._servers[name] = {
                    "status": "error",
                    "error": "server name must be a non-empty string",
                }
                continue
            if not isinstance(raw_spec, dict):
                self._servers[name] = {
                    "status": "error",
                    "error": "server definition must be an object",
                }
                continue
            try:
                spec, material = self._normalize_server(name, raw_spec)
            except MCPConfigError as exc:
                self._servers[name] = {
                    "status": "error",
                    "description": str(raw_spec.get("description") or ""),
                    "error": str(exc),
                }
                continue
            self._specs[name] = spec
            self._fingerprint_material[name] = material
            self._servers[name] = {
                "status": "configured",
                "description": spec.description,
                "transport": spec.transport,
                "tools": [],
            }
        return self.trust_preview

    @property
    def trust_preview(self) -> MCPTrustPreview:
        fingerprint: str | None = None
        if self._fingerprint_material:
            payload = json.dumps(
                self._fingerprint_material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            fingerprint = hashlib.sha256(payload).hexdigest()
        summaries = tuple(
            self._server_preview(self._specs[name]) for name in sorted(self._specs)
        )
        return MCPTrustPreview(
            self.mcp_file.resolve() if self.mcp_file else None,
            fingerprint,
            summaries,
        )

    def mark_untrusted(self) -> None:
        self.load()
        for name in self._specs:
            self._servers[name]["status"] = "pending_trust"
            self._servers[name]["error"] = "configuration has not been trusted"

    async def start(self) -> None:
        self.load()
        await self.runtime.start(spec.connection for spec in self._specs.values())
        self._sync_runtime_states()

    async def start_server(self, name: str, spec: ProjectMCPServer) -> None:
        if name != spec.name:
            raise ValueError("MCP server name does not match its project spec")
        self._specs[name] = spec
        try:
            await self.runtime.start_server(spec.connection)
        finally:
            self._sync_runtime_states()

    async def stop(self) -> None:
        await self.runtime.stop()
        self._servers.clear()
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self._loaded = False

    def server_spec(self, name: str) -> ProjectMCPServer | None:
        self.load()
        return self._specs.get(name)

    def list_tools(self) -> list[str]:
        return self.runtime.list_tools()

    def catalog(self) -> dict[str, dict[str, Any]]:
        return self.runtime.catalog()

    async def call(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float = 60.0,
    ) -> str:
        return await self.runtime.call(server_name, tool_name, args, timeout=timeout)

    def startup_summary(self) -> tuple[str, str]:
        servers = self.servers
        if not servers:
            if self._config_errors:
                return terminal_safe_text(
                    f"MCP: config error — {'; '.join(self._config_errors)}"
                ), "warn"
            return "MCP: none configured", "none"
        connected = [
            name for name, info in servers.items() if info.get("status") == "connected"
        ]
        failed = [
            name for name, info in servers.items() if info.get("status") != "connected"
        ]
        tool_count = len(self.tools)
        prefix = "; ".join(self._config_errors)
        if not failed and not prefix:
            return terminal_safe_text(
                f"MCP: {len(connected)} server(s), {tool_count} tool(s) ({', '.join(connected)})"
            ), "ok"
        details = [
            f"{name}: {servers[name].get('error', servers[name].get('status', 'failed'))}"
            for name in failed
        ]
        if prefix:
            details.insert(0, prefix)
        if connected:
            return terminal_safe_text(
                f"MCP: {len(connected)}/{len(servers)} connected, {tool_count} tool(s) "
                f"({', '.join(connected)}) — {'; '.join(details)}"
            ), "warn"
        return terminal_safe_text(
            f"MCP: 0/{len(servers)} connected — {'; '.join(details)}"
        ), "warn"

    def status(self) -> str:
        lines = [f"Config warning: {error}" for error in self._config_errors]
        servers = self.servers
        if not servers:
            lines.append("No MCP servers configured or started")
            return terminal_safe_text("\n".join(lines), multiline=True)
        for name, info in servers.items():
            status = info.get("status")
            if status == "connected":
                lines.append(f"- {name}: connected ({len(info.get('tools', []))} tools)")
                lines.extend(
                    f"  - mcp__{name}__{tool}" for tool in info.get("tools", [])
                )
            elif status == "pending_trust":
                lines.append(f"- {name}: pending trust")
            elif status == "configured":
                lines.append(f"- {name}: configured, not started")
            elif status == "auth_required":
                lines.append(
                    f"- {name}: authentication required (run `ness mcp login {name}`)"
                )
            else:
                lines.append(f"- {name}: error: {info.get('error', 'failed')}")
        return terminal_safe_text("\n".join(lines), multiline=True)

    async def _runtime_auth(self, spec: MCPServerSpec) -> Any | None:
        if self.http_auth_factory is None:
            return None
        project_spec = self._specs.get(spec.name)
        if project_spec is None:
            raise RuntimeError("MCP project server metadata is unavailable")
        return await self.http_auth_factory(project_spec)

    def _sync_runtime_states(self) -> None:
        for name, state in self.runtime.states.items():
            error = state.error
            if state.status == "auth_required":
                error = f"authentication required; run `ness mcp login {name}`"
            self._servers[name] = {
                "status": state.status,
                "description": state.description,
                "transport": state.transport,
                "tools": list(state.tools),
            }
            if error:
                self._servers[name]["error"] = terminal_safe_text(error)

    def _normalize_server(
        self, name: str, raw: dict[str, Any]
    ) -> tuple[ProjectMCPServer, dict[str, Any]]:
        redactions: list[str] = []
        raw_type = raw.get("type")
        if raw_type is not None and not isinstance(raw_type, str):
            raise MCPConfigError("type must be a string")
        transport = (raw_type or "").lower()
        has_command = "command" in raw
        has_url = "url" in raw
        if has_command and has_url:
            raise MCPConfigError("server cannot contain both command and url")
        if not transport:
            transport = "http" if has_url else "stdio" if has_command else ""
        if transport == "streamable-http":
            transport = "http"
        if transport in {"sse", "ws", "websocket"}:
            raise MCPConfigError(f"unsupported transport: {transport}")
        if transport not in {"stdio", "http"}:
            raise MCPConfigError("type must be stdio, http, or streamable-http")
        if transport == "stdio" and (not has_command or has_url):
            raise MCPConfigError("stdio server requires command and cannot contain url")
        if transport == "http" and (not has_url or has_command):
            raise MCPConfigError("http server requires url and cannot contain command")

        description = raw.get("description", "")
        if not isinstance(description, str):
            raise MCPConfigError("description must be a string")
        timeout = raw.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT)
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise MCPConfigError("startup_timeout must be a positive number")

        if transport == "stdio":
            command_raw = raw.get("command")
            explicit_args = raw.get("args", [])
            if not isinstance(explicit_args, list) or not all(
                isinstance(value, str) for value in explicit_args
            ):
                raise MCPConfigError("args must be an array of strings")
            if isinstance(command_raw, str):
                if not command_raw.strip():
                    raise MCPConfigError("command must not be empty")
                command_parts = [command_raw]
            elif (
                isinstance(command_raw, list)
                and command_raw
                and all(isinstance(value, str) and value for value in command_raw)
            ):
                command_parts = list(command_raw)
            else:
                raise MCPConfigError(
                    "command must be a non-empty string or array of strings"
                )
            command = self._expand(command_parts[0], "command", redactions)
            args = tuple(
                self._expand(value, "args", redactions)
                for value in [*command_parts[1:], *explicit_args]
            )
            raw_cwd = raw.get("cwd")
            if raw_cwd is not None and not isinstance(raw_cwd, str):
                raise MCPConfigError("cwd must be a string")
            cwd = (
                self._resolve_path(self._expand(raw_cwd, "cwd", redactions))
                if raw_cwd
                else self.project_root
            )
            raw_env_file = raw.get("envFile")
            if raw_env_file is not None and not isinstance(raw_env_file, str):
                raise MCPConfigError("envFile must be a string")
            env_file = (
                self._resolve_path(
                    self._expand(raw_env_file, "envFile", redactions)
                )
                if raw_env_file
                else None
            )
            raw_env = _string_mapping(raw.get("env", {}), "env")
            explicit_env = {
                key: self._expand(value, f"env.{key}", redactions)
                for key, value in raw_env.items()
            }
            redactions.extend(explicit_env.values())
            env, env_file_values = self._stdio_environment(env_file, explicit_env)
            redactions.extend(env_file_values)
            connection = MCPServerSpec(
                name=name,
                transport="stdio",
                description=description,
                startup_timeout=float(timeout),
                command=command,
                args=args,
                cwd=cwd,
                env=tuple(sorted(env.items())),
                redactions=tuple(dict.fromkeys(value for value in redactions if value)),
            )
            material = {
                "transport": "stdio",
                "command": command,
                "args": list(args),
                "cwd": str(cwd),
                "envFile": str(env_file) if env_file else None,
                "env": raw_env,
            }
            return ProjectMCPServer(connection=connection, env_file=env_file), material

        raw_url = raw.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            raise MCPConfigError("url must be a non-empty string")
        url = self._expand(raw_url, "url", redactions)
        url_error = validate_mcp_http_url(url)
        if url_error:
            raise MCPConfigError(url_error)
        if raw.get("envFile") is not None:
            raise MCPConfigError("envFile is only supported for stdio servers")
        if raw.get("headersHelper") is not None:
            raise MCPConfigError("headersHelper is not supported")
        raw_headers = _string_mapping(raw.get("headers", {}), "headers")
        headers = {
            key: self._expand(value, f"headers.{key}", redactions)
            for key, value in raw_headers.items()
        }
        redactions.extend(headers.values())
        oauth, raw_oauth = self._normalize_oauth(raw, redactions)
        if oauth is not None and any(key.lower() == "authorization" for key in headers):
            raise MCPConfigError(
                "OAuth cannot be combined with an explicit Authorization header"
            )
        connection = MCPServerSpec(
            name=name,
            transport="http",
            description=description,
            startup_timeout=float(timeout),
            url=url,
            headers=tuple(sorted(headers.items())),
            redactions=tuple(dict.fromkeys(value for value in redactions if value)),
        )
        material = {
            "transport": "http",
            "url": url,
            "headers": raw_headers,
            "oauth": raw_oauth,
        }
        return ProjectMCPServer(connection=connection, oauth=oauth), material

    def _normalize_oauth(
        self, raw: dict[str, Any], redactions: list[str]
    ) -> tuple[MCPOAuthSpec | None, dict[str, Any] | None]:
        cursor_auth = raw.get("auth")
        claude_oauth = raw.get("oauth")
        if cursor_auth is not None and claude_oauth is not None:
            raise MCPConfigError("server cannot contain both auth and oauth")
        if cursor_auth is None and claude_oauth is None:
            return None, None
        if cursor_auth is not None:
            if not isinstance(cursor_auth, dict):
                raise MCPConfigError("auth must be an object")
            client_id_raw = cursor_auth.get("CLIENT_ID")
            if not isinstance(client_id_raw, str) or not client_id_raw:
                raise MCPConfigError("auth.CLIENT_ID must be a non-empty string")
            client_id = self._expand(client_id_raw, "auth.CLIENT_ID", redactions)
            secret_raw = cursor_auth.get("CLIENT_SECRET")
            if secret_raw is not None and not isinstance(secret_raw, str):
                raise MCPConfigError("auth.CLIENT_SECRET must be a string")
            client_secret = (
                self._expand(secret_raw, "auth.CLIENT_SECRET", redactions)
                if secret_raw
                else None
            )
            if client_secret:
                redactions.append(client_secret)
            scopes = self._normalize_scopes(
                cursor_auth.get("scopes", []), "auth.scopes", redactions
            )
            return (
                MCPOAuthSpec(
                    source="cursor",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=scopes,
                    callback_port=8787,
                    token_endpoint_auth_method=(
                        "client_secret_post" if client_secret else "none"
                    ),
                ),
                dict(cursor_auth),
            )
        if not isinstance(claude_oauth, dict):
            raise MCPConfigError("oauth must be an object")
        if claude_oauth.get("authServerMetadataUrl") is not None:
            raise MCPConfigError("oauth.authServerMetadataUrl is not supported")
        client_id_raw = claude_oauth.get("clientId")
        if client_id_raw is not None and not isinstance(client_id_raw, str):
            raise MCPConfigError("oauth.clientId must be a string")
        client_id = (
            self._expand(client_id_raw, "oauth.clientId", redactions)
            if client_id_raw
            else None
        )
        secret_raw = claude_oauth.get("clientSecret")
        if secret_raw is not None and not isinstance(secret_raw, str):
            raise MCPConfigError("oauth.clientSecret must be a string")
        client_secret = (
            self._expand(secret_raw, "oauth.clientSecret", redactions)
            if secret_raw
            else None
        )
        if client_secret:
            redactions.append(client_secret)
        callback_port = claude_oauth.get("callbackPort")
        if callback_port is not None and (
            isinstance(callback_port, bool)
            or not isinstance(callback_port, int)
            or not 1 <= callback_port <= 65535
        ):
            raise MCPConfigError(
                "oauth.callbackPort must be an integer from 1 to 65535"
            )
        scopes = self._normalize_scopes(
            claude_oauth.get("scopes", []), "oauth.scopes", redactions
        )
        default_method = "client_secret_post" if client_secret else "none"
        method = claude_oauth.get("tokenEndpointAuthMethod", default_method)
        if method not in {"none", "client_secret_post", "client_secret_basic"}:
            raise MCPConfigError(
                "oauth.tokenEndpointAuthMethod must be none, client_secret_post, or client_secret_basic"
            )
        return (
            MCPOAuthSpec(
                source="claude",
                client_id=client_id,
                client_secret=client_secret,
                scopes=scopes,
                callback_port=callback_port,
                token_endpoint_auth_method=method,
            ),
            dict(claude_oauth),
        )

    def _normalize_scopes(
        self, value: Any, field: str, redactions: list[str]
    ) -> tuple[str, ...]:
        if isinstance(value, str):
            expanded = self._expand(value, field, redactions)
            return tuple(part for part in expanded.split() if part)
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return tuple(
                scope
                for item in value
                for scope in self._expand(item, field, redactions).split()
                if scope
            )
        raise MCPConfigError(f"{field} must be a string or array of strings")

    def _expand(
        self, value: str, field: str, redactions: list[str] | None = None
    ) -> str:
        reserved = {
            "userHome": str(Path.home()),
            "workspaceFolder": str(self.project_root),
            "workspaceFolderBasename": self.project_root.name,
            "pathSeparator": os.sep,
            "/": os.sep,
        }

        def replace_value(match: re.Match[str]) -> str:
            token = match.group(1)
            if token in reserved:
                return reserved[token]
            if token.startswith("env:"):
                key = token[4:]
                if key and key in os.environ:
                    resolved = os.environ[key]
                    if redactions is not None:
                        redactions.append(resolved)
                    return resolved
                raise MCPConfigError(
                    f"{field} references missing environment variable {key or '<empty>'}"
                )
            if ":-" in token:
                key, default = token.split(":-", 1)
                if not key:
                    raise MCPConfigError(
                        f"{field} contains an invalid environment placeholder"
                    )
                resolved = os.environ.get(key) or default
                if redactions is not None and key in os.environ:
                    redactions.append(resolved)
                return resolved
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
                if token in os.environ:
                    resolved = os.environ[token]
                    if redactions is not None:
                        redactions.append(resolved)
                    return resolved
                raise MCPConfigError(
                    f"{field} references missing environment variable {token}"
                )
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace_value, value)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return (
            path.resolve()
            if path.is_absolute()
            else (self.project_root / path).resolve()
        )

    def _stdio_environment(
        self, env_file: Path | None, explicit_env: dict[str, str]
    ) -> tuple[dict[str, str], list[str]]:
        from mcp.client.stdio import get_default_environment

        result = get_default_environment()
        env_file_values: list[str] = []
        if env_file is not None:
            if not env_file.is_file():
                raise MCPConfigError("envFile does not exist or is not a file")
            try:
                with env_file.open(encoding="utf-8") as handle:
                    if any(binding.error for binding in parse_stream(handle)):
                        raise MCPConfigError("envFile contains invalid syntax")
                parsed = dotenv_values(env_file, interpolate=False)
            except OSError as exc:
                detail = exc.strerror or type(exc).__name__
                raise MCPConfigError(f"cannot read envFile: {detail}") from exc
            parsed_values = {
                key: value for key, value in parsed.items() if value is not None
            }
            result.update(parsed_values)
            env_file_values.extend(parsed_values.values())
        result.update(explicit_env)
        return result, env_file_values

    def _server_preview(self, spec: ProjectMCPServer) -> str:
        if spec.transport == "stdio":
            raw_env = self._fingerprint_material.get(spec.name, {}).get("env", {})
            env_names = sorted(raw_env) if isinstance(raw_env, dict) else []
            command = _redact_text(spec.command or "", spec.redactions)
            args = _redact_text(" ".join(spec.args), spec.redactions)
            cwd = _redact_text(str(spec.cwd), spec.redactions)
            detail = f"{spec.name}: stdio — {command} {args}; cwd={cwd}"
            if spec.env_file:
                detail += (
                    f"; envFile={_redact_text(str(spec.env_file), spec.redactions)}"
                )
            if env_names:
                detail += f"; env keys={', '.join(env_names)}"
            return terminal_safe_text(_redact_text(detail, spec.redactions))
        header_names = ", ".join(sorted(dict(spec.headers))) or "none"
        safe_url = _safe_url(_redact_text(spec.url or "", spec.redactions))
        oauth = ""
        if spec.oauth is not None:
            mode = "static" if spec.oauth.client_id else "dynamic"
            oauth = f"; oauth={mode}"
            if spec.oauth.scopes:
                scopes = _redact_text(", ".join(spec.oauth.scopes), spec.redactions)
                oauth += f" ({scopes})"
        return terminal_safe_text(
            _redact_text(
                f"{spec.name}: http — {safe_url}; header keys={header_names}{oauth}",
                spec.redactions,
            )
        )


def validate_project_mcp_http_url(value: str) -> str | None:
    """Validate a possibly-placeholder-bearing imported project URL."""
    try:
        parts = urlsplit(value)
        _ = parts.port
    except ValueError:
        return "url is malformed"
    if parts.username is not None or parts.password is not None:
        return "url must not contain embedded credentials"
    if parts.fragment:
        return "url must not contain a fragment"
    unresolved = bool(_PLACEHOLDER_RE.search(value))
    if unresolved:
        if parts.scheme and parts.scheme not in {"http", "https"}:
            return "url must be an http(s) URL with a hostname"
        return None
    return validate_mcp_http_url(value)


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str)
        for key, item in value.items()
    ):
        raise MCPConfigError(f"{field} must be an object of string values")
    return dict(value)


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "[invalid URL]"


def _redact_text(
    value: str, secrets: tuple[str, ...], *, fallback: str = "[redacted]"
) -> str:
    result = value
    for secret in sorted(set(secrets), key=len, reverse=True):
        if not secret or secret not in result:
            continue
        if len(secret) < 4:
            return fallback
        result = result.replace(secret, "[redacted]")
    return result
