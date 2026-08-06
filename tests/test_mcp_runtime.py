from __future__ import annotations

import ast
import asyncio
import inspect
from pathlib import Path

import pytest

from ness_agent import MCPRuntime, MCPServerSpec
from ness_agent.mcp import MCPAuthenticationRequired


def test_runtime_connects_without_cli_project_configuration(tmp_path: Path):
    server = Path(__file__).parent / "mcp_echo_server.py"
    runtime = MCPRuntime()
    spec = MCPServerSpec(
        name="echo",
        transport="stdio",
        command="python",
        args=(str(server),),
        cwd=tmp_path,
    )

    async def exercise():
        try:
            await runtime.start([spec])
            assert runtime.states["echo"].status == "connected"
            assert "mcp__echo__echo" in runtime.tools
            result = await runtime.call("echo", "echo", {"message": "domain-neutral"})
            assert result.startswith("domain-neutral")
        finally:
            await runtime.stop()

    asyncio.run(exercise())


def test_runtime_authentication_state_contains_no_cli_guidance():
    async def authentication_required(spec):
        raise MCPAuthenticationRequired("login needed")

    runtime = MCPRuntime(http_auth_factory=authentication_required)
    spec = MCPServerSpec(
        name="remote",
        transport="http",
        url="https://example.com/mcp",
    )

    async def exercise():
        await runtime.start([spec])
        state = runtime.states["remote"]
        assert state.status == "auth_required"
        assert state.error == "authentication required"
        assert "ness" not in state.error.lower()
        await runtime.stop()

    asyncio.run(exercise())


def test_runtime_module_has_no_cli_dependency_or_project_constructor_fields():
    import ness_agent.mcp as mcp_module

    tree = ast.parse(Path(mcp_module.__file__).read_text(encoding="utf-8"))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module or ""
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    )
    assert not any(module.startswith("ness_cli") for module in imported_modules)

    parameters = inspect.signature(MCPRuntime).parameters
    assert "mcp_file" not in parameters
    assert "project_root" not in parameters


def test_runtime_rejects_duplicate_names_before_starting_any_server():
    runtime = MCPRuntime()
    first = MCPServerSpec(name="same", transport="stdio", command="python")
    second = MCPServerSpec(name="same", transport="stdio", command="uv")

    async def exercise():
        with pytest.raises(ValueError, match="duplicate MCP server name"):
            await runtime.start([first, second])
        assert not runtime.states
        assert not runtime.tools

    asyncio.run(exercise())
