from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import AsyncExitStack
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Awaitable, Callable
from typing import Any, Literal
from urllib.parse import urlsplit, urlunsplit

from dotenv import dotenv_values
from dotenv.parser import parse_stream
from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

DEFAULT_STARTUP_TIMEOUT = 20
DEFAULT_CALL_TIMEOUT = 60
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
class MCPServerSpec:
    name: str
    transport: Literal["stdio", "http"]
    description: str
    startup_timeout: float
    command: str | None = None
    args: tuple[str, ...] = ()
    cwd: Path | None = None
    env: tuple[tuple[str, str], ...] = ()
    env_file: Path | None = None
    url: str | None = None
    headers: tuple[tuple[str, str], ...] = ()
    oauth: MCPOAuthSpec | None = None
    redactions: tuple[str, ...] = ()


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


class MCPAuthenticationRequired(RuntimeError):
    """An HTTP server needs an explicit OAuth login."""


HTTPAuthFactory = Callable[[MCPServerSpec], Awaitable[Any | None]]


class MCPManager:
    """Load MCP configuration and expose connected tools as LangChain tools.

    ``load`` is side-effect free: it parses, normalizes, and fingerprints
    configuration but never starts a process or opens a socket. CLI adapters
    use that boundary to enforce project trust before ``start``.

    After ``start``, connected MCP tools are available in ``tools`` under
    names like ``mcp__{server}__{tool}``. Use ``catalog`` for grouped metadata
    and ``call`` for direct invocation.
    """

    def __init__(
        self,
        mcp_file: Path | None = None,
        *,
        project_root: Path = Path.cwd(),
        http_auth_factory: HTTPAuthFactory | None = None,
    ) -> None:
        """Create a manager for one project's MCP configuration.

        Parameters
        ----------
        mcp_file : Path, optional
            Path to the MCP JSON config (typically ``.ness/mcp.json``).
            ``None`` means no file is configured.
        project_root : Path, optional
            Project root used to resolve relative paths and
            ``${workspaceFolder}`` placeholders. Defaults to the current
            working directory.
        """
        self.mcp_file = Path(mcp_file) if mcp_file is not None else None
        self.project_root = Path(project_root).resolve()
        self.http_auth_factory = http_auth_factory
        self.servers: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, Any] = {}
        self.tools: dict[str, StructuredTool] = {}
        self.tool_meta: dict[str, dict[str, Any]] = {}
        self._stacks: dict[str, AsyncExitStack] = {}
        self._server_tasks: dict[str, asyncio.Task[None]] = {}
        self._stop_events: dict[str, asyncio.Event] = {}
        self._specs: dict[str, MCPServerSpec] = {}
        self._fingerprint_material: dict[str, dict[str, Any]] = {}
        self._config_errors: list[str] = []
        self._loaded = False
        self._started = False

    def load(self) -> MCPTrustPreview:
        """Parse and normalize the configured file without connecting servers.

        Idempotent: repeated calls return the cached :class:`MCPTrustPreview`
        without re-reading the file. Populates ``servers`` with per-server
        status (``configured``, ``error``, etc.) and builds internal specs
        used later by ``start``.

        Returns
        -------
        MCPTrustPreview
            Config path, SHA-256 fingerprint, and redacted server summaries
            for trust prompts.
        """
        if self._loaded:
            return self.trust_preview
        self._loaded = True
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self.servers.clear()

        # check if the mcp file exists; check for invalid JSON or OS errors
        if self.mcp_file is None or not self.mcp_file.exists():
            return self.trust_preview
        try:
            config = json.loads(self.mcp_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self._config_errors.append(f"invalid JSON at line {exc.lineno}, column {exc.colno}")
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

        # iterate over the mcpServers and normalize each server
        for raw_name, raw_spec in raw_servers.items():
            name = str(raw_name)
            if not isinstance(raw_name, str) or not raw_name.strip():
                self.servers[name] = {"status": "error", "error": "server name must be a non-empty string"}
                continue
            if not isinstance(raw_spec, dict):
                self.servers[name] = {"status": "error", "error": "server definition must be an object"}
                continue
            try:
                spec, material = self._normalize_server(name, raw_spec)
            except MCPConfigError as exc:
                self.servers[name] = {
                    "status": "error",
                    "description": str(raw_spec.get("description") or ""),
                    "error": str(exc),
                }
                continue
            self._specs[name] = spec
            self._fingerprint_material[name] = material # hashable dict for config
            self.servers[name] = {
                "status": "configured",
                "description": spec.description,
                "transport": spec.transport,
                "tools": [],
            }
        return self.trust_preview

    @property
    def trust_preview(self) -> MCPTrustPreview:
        """Return trust metadata for the loaded configuration.

        The fingerprint is a SHA-256 hash of normalized server material and
        changes when commands, endpoints, env templates, or the server set
        change. Summaries are redacted one-line descriptions suitable for
        interactive approval prompts.

        Returns
        -------
        MCPTrustPreview
            Present even when no config file exists (empty servers, no
            fingerprint).
        """
        fingerprint: str | None = None
        if self._fingerprint_material:
            payload = json.dumps(
                self._fingerprint_material,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            # hash the config to create a fingerprint
            fingerprint = hashlib.sha256(payload).hexdigest()
        # create a summary of the server previews
        summaries = tuple(self._server_preview(self._specs[name]) for name in sorted(self._specs))
        return MCPTrustPreview(self.mcp_file.resolve() if self.mcp_file else None, fingerprint, summaries)

    def mark_untrusted(self) -> None:
        """Mark every loaded server as ``pending_trust`` without disconnecting.

        Ensures configuration is loaded first. Used when the user declines
        an interactive trust prompt or when headless mode skips untrusted
        servers.
        """
        self.load()
        for name in self._specs:
            self.servers[name]["status"] = "pending_trust"
            self.servers[name]["error"] = "configuration has not been trusted"

    async def start(self) -> None:
        """Connect every server from the loaded configuration.

        Loads configuration if needed, then starts all normalized servers
        concurrently. Failures are recorded in ``servers`` with redacted
        error messages; successful servers populate ``tools`` and
        ``sessions``. No-op if already started.
        """
        if self._started:
            return
        self.load()
        self._started = True
        results = await asyncio.gather(
            *(self.start_server(name, spec) for name, spec in self._specs.items()),
            return_exceptions=True,
        )
        for name, result in zip(self._specs, results):
            if isinstance(result, BaseException):
                auth_required = _is_auth_required(result)
                self.servers[name] = {
                    "status": "auth_required" if auth_required else "error",
                    "description": self._specs[name].description,
                    "transport": self._specs[name].transport,
                    "error": (
                        f"authentication required; run `ness mcp login {name}`"
                        if auth_required
                        else self._redact_error(result, self._specs[name])
                    ),
                }

    async def stop(self) -> None:
        """Shut down all MCP servers and reset manager state.

        Signals background tasks to exit, waits for them to finish, and
        clears sessions, tools, specs, and load caches so the manager can
        be reused from a clean slate.
        """
        for event in self._stop_events.values():
            event.set()
        if self._server_tasks:
            await asyncio.gather(*self._server_tasks.values(), return_exceptions=True)
        self._stacks.clear()
        self._server_tasks.clear()
        self._stop_events.clear()
        self.sessions.clear()
        self.tools.clear()
        self.tool_meta.clear()
        self.servers.clear()
        self._specs.clear()
        self._fingerprint_material.clear()
        self._config_errors.clear()
        self._loaded = False
        self._started = False

    def list_tools(self) -> list[str]:
        """Return sorted LangChain tool names exposed by connected servers."""
        return sorted(self.tools)

    def catalog(self) -> dict[str, dict[str, Any]]:
        """Return MCP tools grouped by server for UI and registry metadata.

        Returns
        -------
        dict
            Keys are server names. Each value has ``description`` (from the
            server config) and ``tools`` — a list of dicts with ``name``
            (full LangChain name), ``tool`` (bare MCP name), ``description``,
            and ``arg_names``.
        """
        catalog: dict[str, dict[str, Any]] = {}
        for full_name, meta in self.tool_meta.items():
            server = meta.get("server", "")
            entry = catalog.setdefault(
                server,
                {"description": str(self.servers.get(server, {}).get("description") or ""), "tools": []},
            )
            entry["tools"].append(
                {
                    "name": full_name,
                    "tool": meta.get("tool", ""),
                    "description": meta.get("description", ""),
                    "arg_names": meta.get("arg_names", []),
                }
            )
        return catalog

    def startup_summary(self) -> tuple[str, str]:
        """Return a one-line startup status for CLI banners.

        Returns
        -------
        tuple[str, str]
            ``(message, level)`` where ``level`` is ``"ok"``, ``"warn"``,
            or ``"none"``. Includes config errors, connection counts, and
            failure details when present.
        """
        if not self.servers:
            if self._config_errors:
                return f"MCP: config error — {'; '.join(self._config_errors)}", "warn"
            return "MCP: none configured", "none"
        connected = [n for n, info in self.servers.items() if info.get("status") == "connected"]
        failed = [n for n, info in self.servers.items() if info.get("status") != "connected"]
        tool_count = len(self.tools)
        prefix = "; ".join(self._config_errors)
        if not failed and not prefix:
            return f"MCP: {len(connected)} server(s), {tool_count} tool(s) ({', '.join(connected)})", "ok"
        details = [f"{name}: {self.servers[name].get('error', self.servers[name].get('status', 'failed'))}" for name in failed]
        if prefix:
            details.insert(0, prefix)
        if connected:
            return f"MCP: {len(connected)}/{len(self.servers)} connected, {tool_count} tool(s) ({', '.join(connected)}) — {'; '.join(details)}", "warn"
        return f"MCP: 0/{len(self.servers)} connected — {'; '.join(details)}", "warn"

    def status(self) -> str:
        """Return a multi-line human-readable report of config and server state."""
        lines = [f"Config warning: {error}" for error in self._config_errors]
        if not self.servers:
            lines.append("No MCP servers configured or started")
            return "\n".join(lines)
        for name, info in self.servers.items():
            status = info.get("status")
            if status == "connected":
                lines.append(f"- {name}: connected ({len(info.get('tools', []))} tools)")
                lines.extend(f"  - mcp__{name}__{tool}" for tool in info.get("tools", []))
            elif status == "pending_trust":
                lines.append(f"- {name}: pending trust")
            elif status == "configured":
                lines.append(f"- {name}: configured, not started")
            elif status == "auth_required":
                lines.append(f"- {name}: authentication required (run `ness mcp login {name}`)")
            else:
                lines.append(f"- {name}: error: {info.get('error', 'failed')}")
        return "\n".join(lines)

    async def call(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> str:
        """Invoke one MCP tool on a connected server.

        Parameters
        ----------
        server_name : str
            Configured server name (not the ``mcp__`` LangChain prefix).
        tool_name : str
            Bare tool name as reported by the MCP server.
        args : dict
            Tool arguments matching the server's input schema.
        timeout : float, optional
            Maximum seconds to wait for the call. Defaults to
            ``DEFAULT_CALL_TIMEOUT``.

        Returns
        -------
        str
            Serialized tool output, or an ``"Error: ..."`` string on failure.
            Secrets from the server spec are redacted in error messages.
        """
        session = self.sessions.get(server_name)
        if session is None:
            return f"Error: MCP server not connected: {server_name}"
        payload = _sanitize_tool_args(args)
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, payload), timeout=timeout
            )
        except asyncio.TimeoutError:
            return f"Error: MCP tool call timed out ({timeout}s): {server_name}/{tool_name}"
        except Exception as exc:
            spec = self._specs.get(server_name)
            return f"Error: MCP call failed: {self._redact_error(exc, spec) if spec else type(exc).__name__}"
        if result.isError:
            return "Error: " + _serialize_mcp_result(result)
        return _serialize_mcp_result(result)

    async def start_server(self, name: str, spec: MCPServerSpec | dict[str, Any]) -> None:
        """Connect a single MCP server and register its tools.

        Parameters
        ----------
        name : str
            Server name used as a key in ``servers``, ``sessions``, and tool
            prefixes.
        spec : MCPServerSpec or dict
            Normalized spec or raw server definition (normalized on the fly
            when a dict is passed).

        Raises
        ------
        BaseException
            Propagates connection or initialization failures from the
            background task after ``await`` completes.
        """
        if isinstance(spec, dict):
            spec, _ = self._normalize_server(name, spec)
        if name in self._server_tasks:
            return
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        stop_event = asyncio.Event()
        self._stop_events[name] = stop_event
        self._server_tasks[name] = asyncio.create_task(
            self._run_server(name, spec, ready, stop_event),
            name=f"mcp-server-{name}",
        )
        await ready

    def server_spec(self, name: str) -> MCPServerSpec | None:
        """Return one loaded normalized server specification."""
        self.load()
        return self._specs.get(name)

    async def _run_server(
        self,
        name: str,
        spec: MCPServerSpec,
        ready: asyncio.Future[None],
        stop_event: asyncio.Event,
    ) -> None:
        stack = AsyncExitStack()
        await stack.__aenter__()
        try:
            connect = self._connect_stdio if spec.transport == "stdio" else self._connect_http
            await asyncio.wait_for(connect(name, spec, stack), timeout=spec.startup_timeout)
            self._stacks[name] = stack
            ready.set_result(None)
            await stop_event.wait()
        except BaseException as exc:
            self.sessions.pop(name, None)
            for full_name in [key for key, meta in self.tool_meta.items() if meta.get("server") == name]:
                self.tools.pop(full_name, None)
                self.tool_meta.pop(full_name, None)
            if not ready.done():
                ready.set_exception(exc)
        finally:
            self._stacks.pop(name, None)
            try:
                await stack.aclose()
            except Exception:
                pass

    async def _connect_stdio(self, name: str, spec: MCPServerSpec, stack: AsyncExitStack) -> None:
        from mcp.client.stdio import StdioServerParameters, stdio_client

        params = StdioServerParameters(
            command=spec.command or "",
            args=list(spec.args),
            env=dict(spec.env),
            cwd=str(spec.cwd or self.project_root),
        )
        errlog = open(os.devnull, "w", encoding="utf-8")
        stack.callback(errlog.close)
        read_stream, write_stream = await stack.enter_async_context(stdio_client(params, errlog=errlog))
        await self._initialize_session(name, spec, read_stream, write_stream, stack)

    async def _connect_http(self, name: str, spec: MCPServerSpec, stack: AsyncExitStack) -> None:
        import httpx
        from mcp.client.streamable_http import streamable_http_client

        auth = await self.http_auth_factory(spec) if self.http_auth_factory else None
        client = await stack.enter_async_context(
            httpx.AsyncClient(
                headers=dict(spec.headers),
                auth=auth,
                follow_redirects=True,
                timeout=httpx.Timeout(30.0, read=300.0),
            )
        )
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(spec.url or "", http_client=client)
        )
        await self._initialize_session(name, spec, read_stream, write_stream, stack)

    async def _initialize_session(self, name: str, spec: MCPServerSpec, read_stream: Any, write_stream: Any, stack: AsyncExitStack) -> None:
        from mcp.client.session import ClientSession

        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        result = await session.list_tools()
        self.sessions[name] = session
        self.servers[name] = {
            "status": "connected",
            "description": spec.description,
            "transport": spec.transport,
            "tools": [tool.name for tool in result.tools],
        }
        for mcp_tool in result.tools:
            full_name = f"mcp__{name}__{mcp_tool.name}"
            self.tools[full_name] = self._wrap_tool(name, mcp_tool)
            self.tool_meta[full_name] = {
                "server": name,
                "tool": mcp_tool.name,
                "description": getattr(mcp_tool, "description", "") or "",
                "arg_names": _input_arg_names(getattr(mcp_tool, "inputSchema", None)),
            }

    def _normalize_server(self, name: str, raw: dict[str, Any]) -> tuple[MCPServerSpec, dict[str, Any]]:
        """
        It takes server name and raw JSON object from mcp.json, vlidates it, resolves placeholders,
        and returns two things: a normalized MCPServerSpec and a material dict (used for config fingerprint/trust preview)
        """
        redactions: list[str] = [] # tracks secrets

        raw_type = raw.get("type") # stdio, http, streamable-http
        if raw_type is not None and not isinstance(raw_type, str):
            raise MCPConfigError("type must be a string")
        transport = (raw_type or "").lower()

        # check if 'command' and 'url' exists
        has_command = "command" in raw
        has_url = "url" in raw

        # server cannot be both process based and url-based
        if has_command and has_url:
            raise MCPConfigError("server cannot contain both command and url")

        # if transport is not set, 'http' if 'url' exists, 'stdio' if 'command' exists, otherwise ""
        if not transport:
            transport = "http" if has_url else "stdio" if has_command else ""

        if transport == "streamable-http": # alias for http (legacy)
            transport = "http"

        if transport in {"sse", "ws", "websocket"}: # reject unsupported transports
            raise MCPConfigError(f"unsupported transport: {transport}")

        if transport not in {"stdio", "http"}:
            raise MCPConfigError("type must be stdio, http, or streamable-http")
        if transport == "stdio" and (not has_command or has_url):
            raise MCPConfigError("stdio server requires command and cannot contain url")
        if transport == "http" and (not has_url or has_command):
            raise MCPConfigError("http server requires url and cannot contain command")

        description = raw.get("description", "") # optional description
        if not isinstance(description, str):
            raise MCPConfigError("description must be a string")

        timeout = raw.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT) # startup timeout; default 20s
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or timeout <= 0:
            raise MCPConfigError("startup_timeout must be a positive number")

        if transport == "stdio":
            command_raw = raw.get("command") # string or list of strings ["python", "-u"]
            explicit_args = raw.get("args", []) # must be a list of strings
            if not isinstance(explicit_args, list) or not all(isinstance(value, str) for value in explicit_args):
                raise MCPConfigError("args must be an array of strings")

            # process command
            if isinstance(command_raw, str):
                if not command_raw.strip():
                    raise MCPConfigError("command must not be empty")
                command_parts = [command_raw]
            elif isinstance(command_raw, list) and command_raw and all(isinstance(value, str) and value for value in command_raw):
                command_parts = list(command_raw)
            else:
                raise MCPConfigError("command must be a non-empty string or array of strings")

            command = self._expand(command_parts[0], "command", redactions)
            args = tuple(self._expand(value, "args", redactions) for value in [*command_parts[1:], *explicit_args])

            raw_cwd = raw.get("cwd") # optional working directory
            if raw_cwd is not None and not isinstance(raw_cwd, str):
                raise MCPConfigError("cwd must be a string")
            cwd = self._resolve_path(self._expand(raw_cwd, "cwd", redactions)) if raw_cwd else self.project_root

            raw_env_file = raw.get("envFile") # optional .env file path
            if raw_env_file is not None and not isinstance(raw_env_file, str):
                raise MCPConfigError("envFile must be a string")
            env_file = self._resolve_path(self._expand(raw_env_file, "envFile", redactions)) if raw_env_file else None
            raw_env = _string_mapping(raw.get("env", {}), "env")
            explicit_env = {key: self._expand(value, f"env.{key}", redactions) for key, value in raw_env.items()}
            redactions.extend(explicit_env.values())
            env, env_file_values = self._stdio_environment(env_file, explicit_env) # get the environment variables from the .env file and the explicit environment variables
            redactions.extend(env_file_values)

            spec = MCPServerSpec(
                name=name,
                transport="stdio",
                description=description,
                startup_timeout=float(timeout),
                command=command,
                args=args,
                cwd=cwd,
                env=tuple(sorted(env.items())),
                env_file=env_file,
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
            return spec, material

        # if transport is http, validate and resolve the url
        raw_url = raw.get("url")
        if not isinstance(raw_url, str) or not raw_url:
            raise MCPConfigError("url must be a non-empty string")
        url = self._expand(raw_url, "url", redactions)
        try:
            parts = urlsplit(url)
            _ = parts.port
        except ValueError as exc:
            raise MCPConfigError("url is malformed") from exc

        if parts.scheme not in {"http", "https"} or not parts.hostname:
            raise MCPConfigError("url must be an http(s) URL with a hostname")

        if parts.username is not None or parts.password is not None:
            raise MCPConfigError("url must not contain embedded credentials")

        if parts.fragment:
            raise MCPConfigError("url must not contain a fragment")

        if raw.get("envFile") is not None:
            raise MCPConfigError("envFile is only supported for stdio servers")
        if raw.get("headersHelper") is not None:
            raise MCPConfigError("headersHelper is not supported")
        raw_headers = _string_mapping(raw.get("headers", {}), "headers")
        headers = {key: self._expand(value, f"headers.{key}", redactions) for key, value in raw_headers.items()}
        redactions.extend(headers.values())
        oauth, raw_oauth = self._normalize_oauth(raw, redactions)
        if oauth is not None and any(key.lower() == "authorization" for key in headers):
            raise MCPConfigError("OAuth cannot be combined with an explicit Authorization header")

        spec = MCPServerSpec(
            name=name,
            transport="http",
            description=description,
            startup_timeout=float(timeout),
            url=url,
            headers=tuple(sorted(headers.items())),
            oauth=oauth,
            redactions=tuple(dict.fromkeys(value for value in redactions if value)),
        )
        material = {
            "transport": "http",
            "url": url,
            "headers": raw_headers,
            "oauth": raw_oauth,
        }
        return spec, material

    def _normalize_oauth(
        self,
        raw: dict[str, Any],
        redactions: list[str],
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
            scopes_raw = cursor_auth.get("scopes", [])
            scopes = self._normalize_scopes(scopes_raw, "auth.scopes", redactions)
            method = "client_secret_post" if client_secret else "none"
            return (
                MCPOAuthSpec(
                    source="cursor",
                    client_id=client_id,
                    client_secret=client_secret,
                    scopes=scopes,
                    callback_port=8787,
                    token_endpoint_auth_method=method,
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
            raise MCPConfigError("oauth.callbackPort must be an integer from 1 to 65535")
        scopes = self._normalize_scopes(
            claude_oauth.get("scopes", []),
            "oauth.scopes",
            redactions,
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
        self,
        value: Any,
        field: str,
        redactions: list[str],
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

    def _expand(self, value: str, field: str, redactions: list[str] | None = None) -> str:
        """resolves ${userHome}, ${workspaceFolder}, ${env:VAR}, ${VAR}, and ${VAR:-default}"""
        reserved = {
            "userHome": str(Path.home()),
            "workspaceFolder": str(self.project_root),
            "workspaceFolderBasename": self.project_root.name,
            "pathSeparator": os.sep,
            "/": os.sep,
        }

        def replace(match: re.Match[str]) -> str:
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
                raise MCPConfigError(f"{field} references missing environment variable {key or '<empty>'}")
            if ":-" in token:
                key, default = token.split(":-", 1)
                if not key:
                    raise MCPConfigError(f"{field} contains an invalid environment placeholder")
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
                raise MCPConfigError(f"{field} references missing environment variable {token}")
            return match.group(0)

        return _PLACEHOLDER_RE.sub(replace, value)

    def _resolve_path(self, value: str) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def _stdio_environment(self, env_file: Path | None, explicit_env: dict[str, str]) -> tuple[dict[str, str], list[str]]:
        """
        It takes an optional .env file path and a dictionary of explicit environment variables,
        and returns a tuple of:
        - a dictionary of all environment variables
        - a list of values from the .env file
        """
        from mcp.client.stdio import get_default_environment
        # get_default_environment is similar to os.environ but secure
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
            parsed_values = {key: value for key, value in parsed.items() if value is not None}
            result.update(parsed_values)
            env_file_values.extend(parsed_values.values())
        result.update(explicit_env)
        return result, env_file_values

    def _server_preview(self, spec: MCPServerSpec) -> str:
        if spec.transport == "stdio":
            raw_env = self._fingerprint_material.get(spec.name, {}).get("env", {})
            env_names = sorted(raw_env) if isinstance(raw_env, dict) else []
            command = _redact_text(spec.command or "", spec.redactions)
            args = _redact_text(" ".join(spec.args), spec.redactions)
            cwd = _redact_text(str(spec.cwd), spec.redactions)
            detail = f"{spec.name}: stdio — {command} {args}; cwd={cwd}"
            if spec.env_file:
                detail += f"; envFile={spec.env_file}"
            if env_names:
                detail += f"; env keys={', '.join(env_names)}"
            return detail
        header_names = ", ".join(sorted(dict(spec.headers))) or "none"
        safe_url = _safe_url(_redact_text(spec.url or "", spec.redactions))
        oauth = ""
        if spec.oauth is not None:
            mode = "static" if spec.oauth.client_id else "dynamic"
            oauth = f"; oauth={mode}"
            if spec.oauth.scopes:
                oauth += f" ({', '.join(spec.oauth.scopes)})"
        return f"{spec.name}: http — {safe_url}; header keys={header_names}{oauth}"

    def _redact_error(self, exc: BaseException, spec: MCPServerSpec) -> str:
        message = str(exc) or type(exc).__name__
        if spec.url:
            message = message.replace(spec.url, _safe_url(spec.url))
        return _redact_text(message, spec.redactions, fallback=type(exc).__name__)

    def _wrap_tool(self, server_name: str, mcp_tool: Any) -> StructuredTool:
        full_name = f"mcp__{server_name}__{mcp_tool.name}"
        raw_schema = getattr(mcp_tool, "inputSchema", None)
        args_schema = _args_schema(full_name, raw_schema)
        description = getattr(mcp_tool, "description", "") or f"MCP tool {mcp_tool.name} from {server_name}"
        if isinstance(raw_schema, dict):
            description += f"\n\nInput schema: {json.dumps(raw_schema)}"

        async def _call(**kwargs: Any) -> str:
            return await self.call(server_name, mcp_tool.name, kwargs)

        return StructuredTool.from_function(name=full_name, description=description, coroutine=_call, args_schema=args_schema)


def _string_mapping(value: Any, field: str) -> dict[str, str]:
    if value is None:
        return {}
    if not isinstance(value, dict) or not all(isinstance(key, str) and isinstance(item, str) for key, item in value.items()):
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


def _redact_text(value: str, secrets: tuple[str, ...], *, fallback: str = "[redacted]") -> str:
    result = value
    for secret in sorted(set(secrets), key=len, reverse=True):
        if not secret or secret not in result:
            continue
        if len(secret) < 4:
            return fallback
        result = result.replace(secret, "[redacted]")
    return result


def _is_auth_required(exc: BaseException) -> bool:
    if isinstance(exc, MCPAuthenticationRequired):
        return True
    status_code = getattr(exc, "status_code", None)
    if status_code in {401, 403}:
        return True
    nested = getattr(exc, "exceptions", None)
    if nested and any(_is_auth_required(item) for item in nested):
        return True
    message = str(exc).lower()
    return "401 unauthorized" in message or "authentication required" in message


def _sanitize_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Drop ``null`` optional padding before forwarding tool arguments to MCP.

    Language models frequently set every optional parameter to ``null``. Most
    MCP backends treat omitted fields as "unchanged / default" but reject
    explicit ``null`` for typed inputs (GraphQL scalars, pagination cursors,
    nested filters). Preserves falsy but meaningful values such as ``0``,
    ``False``, and ``""``. Empty nested objects produced only by stripping
    ``null`` children are removed as well.
    """
    if not isinstance(args, dict):
        return {}

    def clean(value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        cleaned: dict[str, Any] = {}
        for key, item in value.items():
            if item is None:
                continue
            normalized = clean(item)
            if normalized == {}:
                continue
            cleaned[key] = normalized
        return cleaned

    cleaned = clean(args)
    return cleaned if isinstance(cleaned, dict) else {}


def _input_arg_names(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    return list(properties.keys()) if isinstance(properties, dict) else []


def _args_schema(name: str, schema: dict[str, Any] | None) -> type:
    if not isinstance(schema, dict):
        return create_model(f"{name}_Args")
    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    fields: dict[str, Any] = {}
    for prop, info in properties.items():
        if not isinstance(info, dict):
            fields[prop] = (Any, None)
            continue
        typ = _resolve_type(info)
        desc = info.get("description", "")
        fields[prop] = (typ, Field(... if prop in required else info.get("default", None), description=desc))
    return create_model(f"{name}_Args", **fields)


def _resolve_type(info: dict[str, Any]) -> type:
    if "enum" in info:
        try:
            from typing import Literal
            return Literal[tuple(info["enum"])]
        except (TypeError, ValueError):
            return str
    return {"object": dict, "array": list, "string": str, "integer": int, "number": float, "boolean": bool}.get(info.get("type"), Any)


def _serialize_mcp_result(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        content_type = getattr(item, "type", None)
        if content_type == "image":
            parts.append("[image]")
        elif content_type == "resource":
            resource = getattr(item, "resource", None)
            parts.append(resource.text if resource and hasattr(resource, "text") else str(item))
        else:
            text = getattr(item, "text", None)
            if text is not None:
                parts.append(text)
            elif hasattr(item, "model_dump"):
                parts.append(json.dumps(item.model_dump(), ensure_ascii=False))
            else:
                parts.append(str(item))
    structured = getattr(result, "structuredContent", None)
    if structured:
        parts.append(json.dumps(structured, ensure_ascii=False))
    return "\n".join(parts) if parts else str(result)


mcp_manager = MCPManager()
