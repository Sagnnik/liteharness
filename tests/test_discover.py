import os
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")

from langchain_core.tools import StructuredTool

import tools as registry
from tools import (
    activate_mcp_tools,
    is_destructive_tool_call,
    is_read_only_tool_call,
    register_dynamic_tools,
    set_mcp_catalog,
    tool_names_for_session,
    tools_generation,
)
from tools.discover import add_tools, search_tools

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


class DiscoverToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self._fake = [_make_tool(name, desc) for name, desc in FAKE_TOOLS.items()]
        register_dynamic_tools(self._fake)
        set_mcp_catalog(CATALOG)
        registry.ACTIVE_MCP_TOOLS.clear()

    def tearDown(self) -> None:
        for name in FAKE_TOOLS:
            registry.TOOL_MAP.pop(name, None)
            registry.MCP_TOOLS.discard(name)
            registry.ACTIVE_MCP_TOOLS.discard(name)
        registry.ALL_TOOLS[:] = [t for t in registry.ALL_TOOLS if t.name not in FAKE_TOOLS]
        registry.TOOL_NAMES[:] = list(registry.TOOL_MAP)
        set_mcp_catalog({})

    def test_search_ranks_relevant_tool_first(self) -> None:
        result = search_tools.invoke({"query": "weather forecast for a city"})
        self.assertIn("mcp__weather__get_forecast", result)
        first_line = [ln for ln in result.splitlines() if ln.startswith("- ")][0]
        self.assertIn("mcp__weather__get_forecast", first_line)

    def test_search_matches_github(self) -> None:
        result = search_tools.invoke({"query": "open a github issue"})
        self.assertIn("mcp__github__create_issue", result)

    def test_search_no_match(self) -> None:
        result = search_tools.invoke({"query": "zzzznonexistentcapability"})
        self.assertIn("No MCP tools matched", result)

    def test_search_excludes_loaded_tools(self) -> None:
        activate_mcp_tools(["mcp__weather__get_forecast"])
        result = search_tools.invoke({"query": "weather forecast for a city"})
        self.assertNotIn("mcp__weather__get_forecast", result)

    def test_add_tools_activates_and_bumps_generation(self) -> None:
        before = tools_generation()
        result = add_tools.invoke({"names": ["mcp__weather__get_forecast"]})
        self.assertIn("Loaded 1 tool", result)
        self.assertIn("mcp__weather__get_forecast", registry.ACTIVE_MCP_TOOLS)
        self.assertGreater(tools_generation(), before)
        # now part of the bound session set
        self.assertIn("mcp__weather__get_forecast", tool_names_for_session())

    def test_add_tools_unknown_name(self) -> None:
        result = add_tools.invoke({"names": ["mcp__nope__missing"]})
        self.assertIn("Unknown", result)

    def test_add_tools_idempotent_no_generation_bump(self) -> None:
        add_tools.invoke({"names": ["mcp__github__create_issue"]})
        gen = tools_generation()
        result = add_tools.invoke({"names": ["mcp__github__create_issue"]})
        self.assertIn("Already loaded", result)
        self.assertEqual(tools_generation(), gen)


if __name__ == "__main__":
    unittest.main()
