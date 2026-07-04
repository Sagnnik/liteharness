from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from tools.ask import question
from tools.discover import add_tools, search_tools
from tools.fs import (
    delete_file,
    edit,
    glob,
    is_git_repo,
    read,
    write,
)
from tools.search import grep
from tools.shell import shell as shell_tool
from tools.subagents import spawn_subagent
from tools.todo import todo
from tools.web import webfetch, web_search

LOCAL_TOOLS = [
    read,
    write,
    delete_file,
    edit,
    glob,
    grep,
    web_search,
    webfetch,
    shell_tool,
    todo,
    search_tools,
    add_tools,
    spawn_subagent,
    question,
]

ALL_TOOLS = list(LOCAL_TOOLS)
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
TOOL_NAMES = list(TOOL_MAP)

SMALL_ALWAYS_ON = {
    "todo",
    "question",
}

TIER_L1 = {
    "read",
    "write",
    "delete_file",
    "edit",
    "grep",
    "web_search",
    "webfetch",
    "glob",
    "shell",
}

TIER_DISCOVERY = {
    "search_tools",
    "add_tools",
}

TIER_L3_ADVANCED = {
    "spawn_subagent",
}

READ_ONLY_TOOLS = {
    "read",
    "grep",
    "web_search",
    "webfetch",
    "glob",
    "todo",
    "search_tools",
    "add_tools",
    "spawn_subagent",
    "question",
}

EDIT_TOOLS = frozenset({
    "write",
    "delete_file",
    "edit",
})

DESTRUCTIVE_TOOLS = set(EDIT_TOOLS) | {
    "shell",
}

# All registered MCP tool names (known catalog). Loaded lazily into the bound set.
MCP_TOOLS: set[str] = set()

# Subset of MCP tools actually bound to the model this session (starts empty;
# grows via add_tools / the /mcp command). Keeping the bound set small preserves
# the provider prefix cache and tool-selection accuracy.
ACTIVE_MCP_TOOLS: set[str] = set()

# Per-server MCP catalog used for search + L1 rendering (set at startup).
# {server: {"description": str, "tools": [{"name", "tool", "description", "arg_names"}]}}
_MCP_CATALOG: dict[str, dict[str, Any]] = {}

# Bumped whenever the bound tool set changes so the agent can hot-rebind mid-session.
_TOOLS_GENERATION = 0

TOOL_CATALOG_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Small always-on", frozenset(SMALL_ALWAYS_ON)),
    ("L1 core", frozenset(TIER_L1)),
    ("Tool discovery", frozenset(TIER_DISCOVERY)),
    ("L3 advanced", frozenset(TIER_L3_ADVANCED)),
)


def catalog_groups_for_render() -> list[tuple[str, set[str]]]:
    """Return tier catalog groups plus the currently loaded MCP tool name set."""
    groups = [(label, set(names)) for label, names in TOOL_CATALOG_GROUPS]
    groups.append(("Loaded MCP tools", set(ACTIVE_MCP_TOOLS)))
    return groups


def tools_generation() -> int:
    """Monotonic counter that changes whenever the bound tool set changes."""
    return _TOOLS_GENERATION


def bump_tools_generation() -> int:
    global _TOOLS_GENERATION
    _TOOLS_GENERATION += 1
    return _TOOLS_GENERATION


def set_mcp_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    """Store the per-server MCP catalog (names + descriptions) for search/rendering."""
    _MCP_CATALOG.clear()
    _MCP_CATALOG.update(catalog or {})


def mcp_catalog() -> dict[str, dict[str, Any]]:
    return _MCP_CATALOG


def activate_mcp_tools(names: Iterable[str]) -> tuple[list[str], list[str]]:
    """Mark MCP tools as active (bound). Returns (added, unknown).

    Bumps the tools generation only when something new was activated so the next
    agent turn hot-rebinds the model with the enlarged tool set.
    """
    added: list[str] = []
    unknown: list[str] = []
    for name in names:
        if name not in TOOL_MAP or not name.startswith("mcp__"):
            unknown.append(name)
            continue
        if name not in ACTIVE_MCP_TOOLS:
            ACTIVE_MCP_TOOLS.add(name)
            added.append(name)
    if added:
        bump_tools_generation()
    return added, unknown


def register_dynamic_tools(tools: Iterable[Any]) -> None:
    for tool in tools:
        if tool.name not in TOOL_MAP:
            ALL_TOOLS.append(tool)
        TOOL_MAP[tool.name] = tool
        if tool.name.startswith("mcp__"):
            MCP_TOOLS.add(tool.name)
    TOOL_NAMES[:] = list(TOOL_MAP)


def get_tools_for_names(names: Iterable[str]) -> list[Any]:
    wanted = set(names)
    return [tool for tool in ALL_TOOLS if tool.name in wanted]


def tool_names_for_session() -> list[str]:
    """Return the currently bound tool set for the current session shape.

    Only MCP tools that have been activated (ACTIVE_MCP_TOOLS) are included, so
    adding MCP servers does not bloat the bound tool set until tools are loaded.
    """
    names = (
        set(SMALL_ALWAYS_ON)
        | set(TIER_L1)
        | set(TIER_DISCOVERY)
        | set(TIER_L3_ADVANCED)
        | set(ACTIVE_MCP_TOOLS)
    )
    return [name for name in TOOL_NAMES if name in names]


def select_tools_for_session(
    tools: Iterable[Any] | None = None,
) -> list[Any]:
    """Select the full stable tool set for a main session."""
    available = list(tools or ALL_TOOLS)
    session_names = set(tool_names_for_session())
    return _dedupe_tools(tool for tool in available if tool.name in session_names)


def is_destructive_tool(name: str) -> bool:
    return name in DESTRUCTIVE_TOOLS or name.startswith("mcp__")


def is_read_only_tool_call(name: str, args: dict[str, Any]) -> bool:
    if name == "shell":
        return _shell_action(args) in {"jobs", "read"}
    return name in READ_ONLY_TOOLS


def is_destructive_tool_call(name: str, args: dict[str, Any]) -> bool:
    if name == "shell":
        return _shell_action(args) in {"run", "start", "kill"}
    return is_destructive_tool(name)


def _shell_action(args: dict[str, Any]) -> str:
    return str(args.get("action") or "").strip().lower()


def _dedupe_tools(tools: Iterable[Any]) -> list[Any]:
    seen: set[str] = set()
    out: list[Any] = []
    for tool in tools:
        name = getattr(tool, "name", "")
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(tool)
    return out
