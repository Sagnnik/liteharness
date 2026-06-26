from __future__ import annotations

import asyncio
import json
from contextlib import AsyncExitStack
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from pydantic import Field, create_model

from config import settings
from permissions import PROJECT_ROOT

MCP_FILE = Path(settings.ness_dir) / "mcp.json"

DEFAULT_STARTUP_TIMEOUT = 20
DEFAULT_CALL_TIMEOUT = 60


class MCPManager:
    """Manage stdio MCP servers and expose their tools as LangChain tools.
    Step 1: main entry point is start() -> start the mcp servers and register the tools
    Step 2: spawn MCP server(s) via stdio -> start_server(name, spec)
    Step 3: initialize + list tools -> _connect_stdio(name, spec, stack)
    Step 4: return StructuredTool wraps over mcp functions (convert to langchain tool) -> _wrap_tool(server_name, mcp_tool)
    Step 5: register_dynamic_tools(..) -> create langchain tools from mcp tools -> register_dynamic_tools()
    Step 6: mcp returns structured content -> convert to string and return -> _serialize_mcp_result(result)
    """

    def __init__(self) -> None:
        # {"server_name": {"status": "connected", "tools": ["tool1", "tool2"]}}
        self.servers: dict[str, dict[str, Any]] = {} 

        # {"server_name": ClientSession}
        self.sessions: dict[str, Any] = {} 

        # {"mcp__{server_name}__{tool_name}": StructuredTool}
        self.tools: dict[str, StructuredTool] = {} 

        # catalog metadata for deferred-tool search/rendering
        # {"mcp__{server}__{tool}": {"server", "tool", "description", "arg_names"}}
        self.tool_meta: dict[str, dict[str, Any]] = {}

        # AsyncExitStack: dynamic container for managing multiple async context managers.
        # easier to maintain and cleanup of mcp servers.
        # {"server_name": AsyncExitStack}
        self._stacks: dict[str, AsyncExitStack] = {} 

        self._started = False # flag to check if the mcp servers are started

    async def start(self) -> None:
        # if the mcp servers are already started, return
        if self._started:
            return
        self._started = True
        if not MCP_FILE.exists():
            return

        try:
            config = json.loads(MCP_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return

        servers = config.get("mcpServers", config.get("servers", {}))
        
        # start the mcp servers in parallel. return exceptions without crashing everything.
        results = await asyncio.gather(*(self.start_server(name, spec) for name, spec in servers.items()), return_exceptions=True)

        # add the results to the servers dictionary.
        for name, result in zip(servers, results):
            if isinstance(result, Exception):
                self.servers[name] = {"status": "error", "error": str(result)}

    async def stop(self) -> None:
        for _, stack in list(self._stacks.items()):
            try:
                await stack.aclose()
            except Exception:
                pass

        self._stacks.clear()
        self.sessions.clear()
        self.tools.clear()
        self.tool_meta.clear()
        self.servers.clear()
        self._started = False

    def list_tools(self) -> list[str]:
        return sorted(self.tools)

    def catalog(self) -> dict[str, dict[str, Any]]:
        """Per-server catalog of tool names + descriptions for deferred-tool search."""
        catalog: dict[str, dict[str, Any]] = {}
        for full_name, meta in self.tool_meta.items():
            server = meta.get("server", "")
            entry = catalog.setdefault(
                server,
                {
                    "description": str(self.servers.get(server, {}).get("description") or ""),
                    "tools": [],
                },
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
        """Return a one-line boot summary: ok, warn, or none."""
        
        # if no servers
        if not self.servers:
            return "MCP: none configured", "none"

        # get the connected and failed servers
        connected = [name for name, info in self.servers.items() if info.get("status") == "connected"]
        failed = [name for name, info in self.servers.items() if info.get("status") != "connected"]
        tool_count = len(self.tools) # total number of tools

        # if no failed servers, return the connected servers and tools
        if not failed:
            names = ", ".join(connected)
            return f"MCP: {len(connected)} server(s), {tool_count} tool(s) ({names})", "ok"

        # if some servers are connected and some are failed, return the connected servers and tools and the failed servers
        if connected:
            names = ", ".join(connected)
            fail_detail = "; ".join(f"{name}: {self.servers[name].get('error', 'failed')}" for name in failed)
            return (
                f"MCP: {len(connected)}/{len(self.servers)} connected, {tool_count} tool(s) ({names}) — {fail_detail}",
                "warn",
            )

        # if no connected servers, return the failed servers
        fail_detail = "; ".join(f"{name}: {self.servers[name].get('error', 'failed')}" for name in failed)
        return f"MCP: 0/{len(self.servers)} connected — {fail_detail}", "warn"

    def status(self) -> str:
        if not self.servers:
            return "No MCP servers configured or started"
        lines: list[str] = []
        for name, info in self.servers.items():
            if info.get("status") == "connected":
                lines.append(f"- {name}: connected ({len(info.get('tools', []))} tools)")
                for tool in info.get("tools", []):
                    lines.append(f"  - mcp__{name}__{tool}")
            else:
                lines.append(f"- {name}: error: {info.get('error')}")
        return "\n".join(lines)

    async def call(
        self,
        server_name: str,
        tool_name: str,
        args: dict[str, Any],
        *,
        timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> str:
        # get the session for the server
        session = self.sessions.get(server_name)
        if session is None:
            return f"Error: MCP server not connected: {server_name}"
        try:
            # set tool call timeout with asyncio.wait_for
            result = await asyncio.wait_for(
                session.call_tool(tool_name, args),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            return f"Error: MCP tool call timed out ({timeout}s): {server_name}/{tool_name}"
        except Exception as exc:
            return f"Error: MCP call failed: {exc}"

        if result.isError:
            return "Error: " + _serialize_mcp_result(result)
        return _serialize_mcp_result(result)

    async def start_server(self, name: str, spec: dict[str, Any]) -> None:
        # get the timeout 
        startup_timeout = int(spec.get("startup_timeout", DEFAULT_STARTUP_TIMEOUT))
        
        # create a new stack for each server.
        stack = AsyncExitStack()
        await stack.__aenter__() # init

        # connect to the stdio server.
        try:
            await asyncio.wait_for(
                self._connect_stdio(name, spec, stack),
                timeout=startup_timeout,
            )
        except Exception:
            await stack.aclose()
            raise
        # add the stack to the stacks dictionary.
        self._stacks[name] = stack

    async def _connect_stdio(
        self,
        name: str,
        spec: dict[str, Any],
        stack: AsyncExitStack,
    ) -> None:
        from mcp.client.session import ClientSession
        from mcp.client.stdio import StdioServerParameters, stdio_client

        #parse the command and args from the spec.
        command, args = _command_and_args(spec)

        # create the stdio server parameters. 
        # requires command, args (could be []). env and cwd are dependent on the spec.
        params = StdioServerParameters(
            command=command,
            args=args,
            env=spec.get("env"),
            cwd=spec.get("cwd") or str(PROJECT_ROOT),
        )

        # stdio_client yields two streams: read_stream (from server) and write_stream (to server).
        # put them inside async context manager or context stack 
        read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
        
        # ClientSession consumes those raw streams and provides session object with async methods like call_tool, list_tools, etc.
        # put this inside async context manager or context stack as well
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        
        # do the mcp handshake
        await session.initialize()
        
        self.sessions[name] = session
        result = await session.list_tools() # list tools in a server
        self.servers[name] = {
            "status": "connected",
            "description": str(spec.get("description") or ""),
            "tools": [tool.name for tool in result.tools],
        }

        # wrap the tools for langchain tool
        for mcp_tool in result.tools:
            full_name = f"mcp__{name}__{mcp_tool.name}"
            self.tools[full_name] = self._wrap_tool(name, mcp_tool)
            self.tool_meta[full_name] = {
                "server": name,
                "tool": mcp_tool.name,
                "description": getattr(mcp_tool, "description", "") or "",
                "arg_names": _input_arg_names(getattr(mcp_tool, "inputSchema", None)),
            }

    def _wrap_tool(self, server_name: str, mcp_tool: Any) -> StructuredTool:
        """Translate MCP tool to LangChain tool (StructuredTool)"""
        
        full_name = f"mcp__{server_name}__{mcp_tool.name}"
        raw_schema = getattr(mcp_tool, "inputSchema", None)
        args_schema = _args_schema(full_name, raw_schema)
        description = getattr(mcp_tool, "description", "") or f"MCP tool {mcp_tool.name} from {server_name}"
        if isinstance(raw_schema, dict):
            description += f"\n\nInput schema: {json.dumps(raw_schema)}"

        # define the _call function that will be executed when the tool is invoked (await tool.ainvoke()).
        async def _call(**kwargs: Any) -> str:
            # whatever LLM generates is sent as kwargs
            return await self.call(server_name, mcp_tool.name, kwargs) 


        return StructuredTool.from_function(
            name=full_name,
            description=description,
            coroutine=_call, # marks the tool as a coroutine to be executed asynchronously
            args_schema=args_schema,
        )


def _input_arg_names(schema: dict[str, Any] | None) -> list[str]:
    if not isinstance(schema, dict):
        return []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return []
    return list(properties.keys())


def _command_and_args(spec: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Example:
    {
        "command": "uv",
        "args": ["run", "python", "tests/mcp_echo_server.py"],
    }
    or,
    {
        "command" : ["uv", "run", "python", "tests/mcp_echo_server.py"]
    }
    """
    command = spec.get("command")
    args = spec.get("args", [])
    if isinstance(command, list):
        return str(command[0]), [str(item) for item in command[1:]]
    return str(command), [str(item) for item in args]


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
        if prop in required:
            fields[prop] = (typ, Field(..., description=desc))
        else:
            default = info.get("default", None)
            fields[prop] = (typ, Field(default=default, description=desc))
    return create_model(f"{name}_Args", **fields)


def _resolve_type(info: dict[str, Any]) -> type:
    if "enum" in info:
        try:
            from typing import Literal

            return Literal[tuple(info["enum"])]
        except (TypeError, ValueError):
            return str
    json_type = info.get("type")
    if json_type == "object":
        return dict
    if json_type == "array":
        return list
    return {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
    }.get(json_type, Any)


def _serialize_mcp_result(result: Any) -> str:
    parts: list[str] = []
    for item in getattr(result, "content", []) or []:
        content_type = getattr(item, "type", None)
        if content_type == "image":
            parts.append("[image]")
        elif content_type == "resource":
            resource = getattr(item, "resource", None)
            if resource and hasattr(resource, "text"):
                parts.append(resource.text)
            else:
                parts.append(str(item))
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
