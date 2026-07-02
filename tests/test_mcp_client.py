"""Integration tests for mcp_client.py only."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_client import MCPManager


async def _exercise_manager(mgr: MCPManager) -> None:
    await mgr.start()

    assert "echo" in mgr.servers, mgr.status()
    assert mgr.servers["echo"]["status"] == "connected", mgr.status()
    assert mgr.servers["echo"]["tools"] == ["echo"]

    tool_names = mgr.list_tools()
    assert "mcp__echo__echo" in tool_names

    direct = await mgr.call("echo", "echo", {"message": "hello"})
    assert "hello" in direct

    wrapped = await mgr.tools["mcp__echo__echo"].ainvoke({"message": "via-wrapper"})
    assert "via-wrapper" in str(wrapped)

    status = mgr.status()
    assert "echo: connected (1 tools)" in status
    assert "mcp__echo__echo" in status

    summary, level = mgr.startup_summary()
    assert level == "ok"
    assert "1 server(s), 1 tool(s)" in summary
    assert "echo" in summary

    catalog = mgr.catalog()
    assert "echo" in catalog
    echo_tools = catalog["echo"]["tools"]
    assert echo_tools[0]["name"] == "mcp__echo__echo"
    assert echo_tools[0]["tool"] == "echo"

    await mgr.stop()
    assert not mgr.tool_meta
    assert not mgr.sessions
    assert not mgr.tools
    assert not mgr.servers


def test_mcp_manager_with_echo_server() -> None:
    asyncio.run(_exercise_manager(MCPManager()))


def test_startup_summary_when_unconfigured() -> None:
    mgr = MCPManager()
    message, level = mgr.startup_summary()
    assert level == "none"
    assert "none configured" in message


def test_singleton_start_is_idempotent() -> None:
    async def run() -> None:
        fresh = MCPManager()
        await fresh.start()
        await fresh.start()
        assert "echo" in fresh.servers
        await fresh.stop()

    asyncio.run(run())
