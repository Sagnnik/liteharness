"""SDK Phase B: discover/add_tools + ToolRegistry MCP lifecycle."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.tools import StructuredTool

from liteharness.options import NessAgentOptions
from liteharness.permissions import PermissionStore
from liteharness.persistence import ThreadStore
from liteharness.session_context import SessionContext, set_session_context
from liteharness.tools import ToolRegistry
from liteharness.tools.discover import add_tools, search_tools


FAKE_TOOLS = {
    "mcp__weather__get_forecast": "Get the weather forecast for a city",
    "mcp__github__create_issue": "Create a github issue with a title and body",
    "mcp__github__list_pulls": "List github pull requests for a repository",
}

CATALOG = {
    "weather": {
        "description": "weather forecasts and conditions",
        "tools": [
            {
                "name": "mcp__weather__get_forecast",
                "tool": "get_forecast",
                "description": "Get the weather forecast for a city",
                "arg_names": ["city", "days"],
            }
        ],
    },
    "github": {
        "description": "github issues and pull requests",
        "tools": [
            {
                "name": "mcp__github__create_issue",
                "tool": "create_issue",
                "description": "Create a github issue with a title and body",
                "arg_names": ["title", "body"],
            },
            {
                "name": "mcp__github__list_pulls",
                "tool": "list_pulls",
                "description": "List github pull requests for a repository",
                "arg_names": ["repo"],
            },
        ],
    },
}


def _make_tool(name: str, description: str) -> StructuredTool:
    def fn() -> str:
        return "ok"

    return StructuredTool.from_function(func=fn, name=name, description=description)


def _install_registry(tmp_path: Path, reg: ToolRegistry) -> ToolRegistry:
    ness = tmp_path / ".ness"
    ness.mkdir(parents=True, exist_ok=True)
    options = NessAgentOptions(project_root=tmp_path, ness_dir=ness)
    permission_store = PermissionStore(ness_dir=ness, project_root=tmp_path)
    store = ThreadStore(threads_dir=ness / "threads", auto_save=False)
    cfg = SimpleNamespace(tool_registry=reg)
    set_session_context(
        SessionContext(
            permissions=permission_store,
            options=options,
            thread_store=store,
            ness_dir=ness,
            project_root=tmp_path,
            agent_config=cfg,  # type: ignore[arg-type]
        )
    )
    return reg


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    reg = ToolRegistry(include=["search_tools", "add_tools", "todo"])
    fake = [_make_tool(name, desc) for name, desc in FAKE_TOOLS.items()]
    reg.register_dynamic(fake)
    reg.set_mcp_catalog(CATALOG)
    return _install_registry(tmp_path, reg)


def test_search_ranks_relevant_tool_first(registry: ToolRegistry):
    result = search_tools.invoke({"query": "weather forecast for a city"})
    assert "mcp__weather__get_forecast" in result
    first_line = [ln for ln in result.splitlines() if ln.startswith("- ")][0]
    assert "mcp__weather__get_forecast" in first_line


def test_search_matches_github(registry: ToolRegistry):
    result = search_tools.invoke({"query": "open a github issue"})
    assert "mcp__github__create_issue" in result


def test_search_no_match(registry: ToolRegistry):
    result = search_tools.invoke({"query": "zzzznonexistentcapability"})
    assert "No MCP tools matched" in result


def test_search_excludes_loaded_tools(registry: ToolRegistry):
    registry.activate_mcp(["mcp__weather__get_forecast"])
    result = search_tools.invoke({"query": "weather forecast for a city"})
    assert "mcp__weather__get_forecast" not in result


def test_add_tools_activates_on_registry(registry: ToolRegistry):
    before = registry.generation()
    result = add_tools.invoke({"names": ["mcp__weather__get_forecast"]})
    assert "Loaded 1 tool" in result
    assert "mcp__weather__get_forecast" in registry.active_mcp_tools
    assert registry.generation() > before
    assert "mcp__weather__get_forecast" in registry.tool_names()


def test_add_tools_unknown_name(registry: ToolRegistry):
    result = add_tools.invoke({"names": ["mcp__nope__missing"]})
    assert "Unknown" in result


def test_add_tools_idempotent_no_generation_bump(registry: ToolRegistry):
    add_tools.invoke({"names": ["mcp__github__create_issue"]})
    gen = registry.generation()
    result = add_tools.invoke({"names": ["mcp__github__create_issue"]})
    assert "Already loaded" in result
    assert registry.generation() == gen


def test_deactivate_mcp_round_trip(registry: ToolRegistry):
    added, unknown = registry.activate_mcp(["mcp__weather__get_forecast"])
    assert added == ["mcp__weather__get_forecast"]
    assert unknown == []
    assert "mcp__weather__get_forecast" in registry.tool_names()

    removed, unknown2 = registry.deactivate_mcp(["mcp__weather__get_forecast"])
    assert removed == ["mcp__weather__get_forecast"]
    assert unknown2 == []
    assert "mcp__weather__get_forecast" not in registry.active_mcp_tools
    assert "mcp__weather__get_forecast" not in registry.tool_names()
    # Still registered for re-activation
    assert "mcp__weather__get_forecast" in registry._tool_map

    re_added, _ = registry.activate_mcp(["mcp__weather__get_forecast"])
    assert re_added == ["mcp__weather__get_forecast"]


def test_deactivate_unknown(registry: ToolRegistry):
    removed, unknown = registry.deactivate_mcp(["mcp__nope__missing"])
    assert removed == []
    assert unknown == ["mcp__nope__missing"]


def test_dedupe_drops_duplicate_names():
    t1 = _make_tool("read", "first")
    t2 = _make_tool("read", "second")
    t3 = _make_tool("todo", "todo")
    reg = ToolRegistry([t1, t2, t3], include=["read", "todo"])
    names = [t.name for t in reg.active_tools]
    assert names.count("read") == 1
    assert "todo" in names


def test_tool_names_sorted():
    reg = ToolRegistry(include=["todo", "shell", "read"])
    assert reg.tool_names() == sorted(reg.tool_names())
    assert reg.tool_names() == ["read", "shell", "todo"]
