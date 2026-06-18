from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from langchain_core.tools import StructuredTool

from tools.check_syntax import check_syntax
from tools.fs import (
    apply_patch,
    delete_file,
    edit_file,
    glob_files,
    is_git_repo,
    list_files,
    multi_edit,
    read_file,
    write_file,
)
from tools.git import (
    git_branch,
    git_checkout,
    git_commit,
    git_diff,
    git_log,
    git_show,
    git_stash,
    git_status,
)
from tools.search import grep
from tools.shell import shell as shell_tool
from tools.subagents import spawn_subagent
from tools.todo import todo_read, todo_write
from tools.web import fetch_url, web_search

def _load_project_context() -> str:
    from memory import load_repo_context

    return load_repo_context()


get_project_context = StructuredTool.from_function(
    name="get_project_context",
    description="Return compact project structure and key manifest snippets.",
    func=_load_project_context,
)

LOCAL_TOOLS = [
    read_file,
    write_file,
    delete_file,
    edit_file,
    multi_edit,
    apply_patch,
    glob_files,
    list_files,
    grep,
    check_syntax,
    web_search,
    fetch_url,
    shell_tool,
    git_status,
    git_diff,
    git_log,
    git_show,
    git_commit,
    git_checkout,
    git_branch,
    git_stash,
    todo_write,
    todo_read,
    get_project_context,
    spawn_subagent,
]

ALL_TOOLS = list(LOCAL_TOOLS)
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
TOOL_NAMES = list(TOOL_MAP)

SMALL_ALWAYS_ON = {
    "todo_read",
    "todo_write",
}

TIER_L1 = {
    "read_file",
    "write_file",
    "delete_file",
    "edit_file",
    "multi_edit",
    "apply_patch",
    "grep",
    "check_syntax",
    "web_search",
    "fetch_url",
    "glob_files",
    "list_files",
    "shell",
    "get_project_context",
}

TIER_L2_GIT = {
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
}

TIER_L3_GIT = {
    "git_commit",
    "git_checkout",
    "git_branch",
    "git_stash",
}

TIER_L3_ADVANCED = {
    "spawn_subagent",
}

GIT_TOOLS = TIER_L2_GIT | TIER_L3_GIT

READ_ONLY_TOOLS = {
    "read_file",
    "grep",
    "check_syntax",
    "web_search",
    "fetch_url",
    "glob_files",
    "list_files",
    "git_status",
    "git_diff",
    "git_log",
    "git_show",
    "todo_read",
    "get_project_context",
    "todo_write",
    "spawn_subagent",
}

PLAN_MODE_TOOLS = READ_ONLY_TOOLS | {
    "shell",
}

EDIT_TOOLS = frozenset({
    "write_file",
    "delete_file",
    "edit_file",
    "multi_edit",
    "apply_patch",
})

DESTRUCTIVE_TOOLS = set(EDIT_TOOLS) | {
    "shell",
    "git_commit",
    "git_checkout",
    "git_branch",
    "git_stash",
}

MCP_TOOLS: set[str] = set()

TOOL_CATALOG_GROUPS: tuple[tuple[str, frozenset[str]], ...] = (
    ("Small always-on", frozenset(SMALL_ALWAYS_ON)),
    ("L1 core", frozenset(TIER_L1)),
    ("L2 git read", frozenset(TIER_L2_GIT)),
    ("L3 git write", frozenset(TIER_L3_GIT)),
    ("L3 advanced", frozenset(TIER_L3_ADVANCED)),
)


def catalog_groups_for_render() -> list[tuple[str, set[str]]]:
    """Return tier catalog groups plus the current MCP tool name set."""
    groups = [(label, set(names)) for label, names in TOOL_CATALOG_GROUPS]
    groups.append(("Dynamic MCP", set(MCP_TOOLS)))
    return groups


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


def tool_names_for_mode(agent_mode: str = "normal", git_repo: bool | None = None) -> list[str]:
    """Return the stable tool tier names for a mode."""
    git_available = is_git_repo() if git_repo is None else git_repo
    if agent_mode == "plan":
        names = set(PLAN_MODE_TOOLS)
    else:
        names = set(SMALL_ALWAYS_ON) | set(TIER_L1) | set(TIER_L3_ADVANCED)
    if git_available:
        if agent_mode != "plan":
            names |= set(GIT_TOOLS)
    else:
        names -= set(GIT_TOOLS)
    return [name for name in TOOL_NAMES if name in names]


def tool_names_for_session(git_repo: bool | None = None) -> list[str]:
    """Return the full stable tool set for the current session shape."""
    git_available = is_git_repo() if git_repo is None else git_repo
    names = set(SMALL_ALWAYS_ON) | set(TIER_L1) | set(TIER_L3_ADVANCED) | set(MCP_TOOLS)
    if git_available:
        names |= set(GIT_TOOLS)
    else:
        names -= set(GIT_TOOLS)
    return [name for name in TOOL_NAMES if name in names]


def select_tools_for_session(
    git_repo: bool | None = None,
    tools: Iterable[Any] | None = None,
) -> list[Any]:
    """Select the full stable tool set for a main session."""
    available = list(tools or ALL_TOOLS)
    session_names = set(tool_names_for_session(git_repo))
    return _dedupe_tools(tool for tool in available if tool.name in session_names)


def select_tools_for_mode(
    agent_mode: str = "normal",
    git_repo: bool | None = None,
    tools: Iterable[Any] | None = None,
) -> list[Any]:
    """Select native tools for the current mode and repository shape."""
    available = list(tools or ALL_TOOLS)
    tier_names = set(tool_names_for_mode(agent_mode, git_repo))
    selected = [tool for tool in available if tool.name in tier_names]
    if agent_mode != "plan":
        selected.extend(tool for tool in available if tool.name in MCP_TOOLS)
    return _dedupe_tools(selected)


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
