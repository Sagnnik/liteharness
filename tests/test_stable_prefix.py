"""Stable-prefix cache + deferred MCP L1 rendering."""

from __future__ import annotations

from liteharness.context.layers import PromptLayers, PromptLayersConfig
from liteharness.tools import ToolRegistry


def _layers() -> PromptLayers:
    return PromptLayers(PromptLayersConfig(l0="L0 harness.", persona="Tester."))


def test_skill_catalog_change_busts_stable_prefix_cache():
    layers = _layers()
    kwargs = dict(
        tools=[],
        user_memory="",
        project_memory="",
        git_available=False,
        metadata={},
    )
    first = layers.build_stable_prefix(**kwargs, skill_catalog="- skill-a: does a")
    second = layers.build_stable_prefix(**kwargs, skill_catalog="- skill-b: does b")
    assert first != second
    assert "skill-a" in first
    assert "skill-b" in second
    # Same catalog again hits cache (content identical).
    again = layers.build_stable_prefix(**kwargs, skill_catalog="- skill-b: does b")
    assert again == second
    assert layers._cache.get("content") == second


def test_deferred_mcp_rendered_into_l1_with_header_not_full_tool_dump():
    reg = ToolRegistry(include=["todo"])
    # Many deferred tools — summary must stay per-server, not enumerate all.
    tools = [
        {
            "name": f"mcp__bulk__tool_{i}",
            "tool": f"tool_{i}",
            "description": f"tool {i}",
            "arg_names": [],
        }
        for i in range(50)
    ]
    reg.set_mcp_catalog(
        {
            "weather": {
                "description": "weather forecasts and conditions",
                "tools": [
                    {
                        "name": "mcp__weather__get_forecast",
                        "tool": "get_forecast",
                        "description": "forecast",
                        "arg_names": ["city"],
                    }
                ],
            },
            "bulk": {"description": "", "tools": tools},
        }
    )
    summary = reg.deferred_mcp_summary()
    assert "use search_tools to find, add_tools to load" in summary
    assert "mcp__weather__*" in summary
    assert "1 tool(s)" in summary
    assert "weather forecasts" in summary
    assert "mcp__bulk__*" in summary
    assert "50 tool(s)" in summary
    # Sample tool names when no server description — not all 50 names.
    assert "tool_0" in summary
    assert summary.count("mcp__bulk__tool_") == 0  # full names not dumped

    layers = _layers()
    prefix = layers.build_stable_prefix(
        [],
        user_memory="",
        project_memory="",
        skill_catalog="",
        git_available=False,
        metadata={},
        deferred_mcp=summary,
    )
    assert "Available MCP servers" in prefix
    assert "mcp__weather__*" in prefix


def test_empty_mcp_catalog_omits_deferred_section():
    reg = ToolRegistry(include=["todo"])
    assert reg.deferred_mcp_summary() == ""
    layers = _layers()
    prefix = layers.build_stable_prefix(
        [],
        user_memory="",
        project_memory="",
        skill_catalog="",
        git_available=False,
        metadata={},
        deferred_mcp="",
    )
    assert "Available MCP servers" not in prefix
