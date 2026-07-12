from __future__ import annotations
from collections.abc import Iterable
from typing import Any
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool

from liteharness.tools.ask import question
from liteharness.tools.discover import add_tools, search_tools
from liteharness.tools.fs import delete_file, edit, glob, is_git_repo, read, write
from liteharness.tools.search import grep
from liteharness.tools.shell import shell as shell_tool
from liteharness.tools.skill import skill_view
from liteharness.tools.subagents import spawn_subagent
from liteharness.tools.todo import todo
from liteharness.tools.web import webfetch, web_search

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
    skill_view,
]

ALL_TOOLS = list(LOCAL_TOOLS)
TOOL_MAP = {tool.name: tool for tool in ALL_TOOLS}
TOOL_NAMES = list(TOOL_MAP)

SMALL_ALWAYS_ON = {"todo", "question", "skill_view"}
TIER_L1 = {"read", "write", "delete_file", "edit", "grep", "web_search", "webfetch", "glob", "shell"}
TIER_DISCOVERY = {"search_tools", "add_tools"}
TIER_L3_ADVANCED = {"spawn_subagent"}
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
    "skill_view",
}

EDIT_TOOLS = frozenset({"write", "delete_file", "edit"})
DESTRUCTIVE_TOOLS = set(EDIT_TOOLS) | {"shell"}

TOOL_CATALOG_GROUPS = (
    ("Small always-on", frozenset(SMALL_ALWAYS_ON)),
    ("L1 core", frozenset(TIER_L1)),
    ("Tool discovery", frozenset(TIER_DISCOVERY)),
    ("L3 advanced", frozenset(TIER_L3_ADVANCED)),
)
FULL_TOOL_SET = set(SMALL_ALWAYS_ON) | set(TIER_L1) | set(TIER_DISCOVERY) | set(TIER_L3_ADVANCED)

# Module-level MCP catalog used by search_tools / add_tools until a session
# ToolRegistry is the sole source of truth (Phase B).
ACTIVE_MCP_TOOLS: set[str] = set()
_MCP_CATALOG: dict[str, dict[str, Any]] = {}


def mcp_catalog() -> dict[str, dict[str, Any]]:
    return _MCP_CATALOG


def set_mcp_catalog(catalog: dict[str, dict[str, Any]]) -> None:
    _MCP_CATALOG.clear()
    _MCP_CATALOG.update(catalog or {})


def activate_mcp_tools(names: Iterable[str]) -> tuple[list[str], list[str]]:
    added: list[str] = []
    unknown: list[str] = []
    for name in names:
        if name not in TOOL_MAP or not name.startswith("mcp__"):
            unknown.append(name)
            continue
        if name not in ACTIVE_MCP_TOOLS:
            ACTIVE_MCP_TOOLS.add(name)
            added.append(name)
    return added, unknown


def tool_names_for_session() -> list[str]:
    return list(FULL_TOOL_SET | set(ACTIVE_MCP_TOOLS))


class ToolRegistry:
    """Bound tool set with optional MCP hot-rebind."""

    def __init__(
        self, 
        tools: Iterable[BaseTool] | None = None,
        *, 
        include: Iterable[str] | None = None,
    ) -> None:
        """Bind a set of tools, optionally filtering by name.

        Args:
            tools: Tool instances. When ``None``, defaults to :const:`LOCAL_TOOLS`.
            include: If given, only tools whose names appear in this iterable
                     are activated. The full set remains available for later
                     activation via :meth:`activate_mcp` or :meth:`register_dynamic`.
        """
        self._all_tools: list[BaseTool] = list(tools) if tools is not None else list(LOCAL_TOOLS)
        self._include: set[str] | None = set(include) if include else None
        self._tool_map: dict[str, BaseTool] = {t.name: t for t in self._all_tools}
        self._mcp_catalog: dict[str, dict[str, Any]] = {}
        self.active_mcp_tools: set[str] = set()
        self._generation = 0
        self.runtime: dict[str, Any] = {}
        self._sync(force=True)

    def _sync(self, force: bool = False) -> None:
        if self.runtime and not force and self.runtime.get("generation") == self._generation:
            return
        if self._include is not None:
            active = [t for t in self._all_tools if t.name in self._include]
        else:
            active = [t for t in self._all_tools if t.name in FULL_TOOL_SET]
        active = self._dedupe(active)
        self.runtime["active_tools"] = active
        self.runtime["tool_map"] = {t.name: t for t in active}
        self.runtime["tool_names"] = list(self.runtime["tool_map"])
        self.runtime["generation"] = self._generation

    @property
    def active_tools(self) -> list[BaseTool]:
        """The current list of active tool instances (lazy-synced)."""
        self._sync()
        return self.runtime["active_tools"]

    def tool_map(self) -> dict[str, BaseTool]:
        """Map of tool name → tool instance for active tools (lazy-synced)."""
        self._sync()
        return self.runtime["tool_map"]

    def tool_names(self) -> list[str]:
        """Sorted list of active tool names (lazy-synced)."""
        self._sync()
        return self.runtime["tool_names"]

    def bind_model(self, model: BaseChatModel) -> BaseChatModel:
        """Bind the currently active tools to *model* and return it.

        The returned model is ready for use with langgraph.
        """
        self._sync()
        return model.bind_tools(self.runtime["active_tools"])

    def sync(self) -> None:
        """Force a re-synchronisation of the active tool set."""
        self._sync(force=True)

    def generation(self) -> int:
        """Current generation counter — incremented on every structural change."""
        return self._generation

    def bump_generation(self) -> int:
        """Force-increment the generation counter (e.g. after a dynamic update)."""
        self._generation += 1
        return self._generation

    def tool_catalog_groups(self) -> list[tuple[str, set[str]]]:
        """Return tiered tool groupings for prompt rendering.

        Each entry is ``(label, set_of_tool_names)``. Groups with no
        active tools are omitted.
        """
        groups = [
            (label, set(names) & set(self.runtime["tool_names"]))
            for label, names in TOOL_CATALOG_GROUPS
        ]
        groups.append(("Loaded MCP tools", set(self.active_mcp_tools) & set(self.runtime["tool_names"])))
        return [(l, g) for l, g in groups if g]

    def mcp_catalog(self) -> dict[str, dict[str, Any]]:
        """The full MCP server catalog loaded by the session."""
        return self._mcp_catalog

    def set_mcp_catalog(self, catalog: dict[str, dict[str, Any]] | None) -> None:
        """Replace the MCP catalog (clears previous entries)."""
        self._mcp_catalog.clear()
        self._mcp_catalog.update(catalog or {})

    def deferred_mcp_summary(self) -> str:
        """Human-readable summary of MCP tools not yet activated."""
        if not self._mcp_catalog:
            return ""
        lines = ["- Available MCP servers (use search_tools to find, add_tools to load):"]
        for server in sorted(self._mcp_catalog):
            info = self._mcp_catalog[server]
            count = sum(1 for e in info.get("tools", []) if e.get("name") not in self.active_mcp_tools)
            if count == 0:
                continue
            desc = str(info.get("description") or "").strip().replace("\n", " ")[:100]
            lines.append(f"  - mcp__{server}__* ({count} tool(s)){': ' + desc if desc else ''}")
        return "\n".join(lines[1:]) if len(lines) > 1 else ""

    def register_dynamic(self, tools: Iterable[BaseTool]) -> None:
        """Register dynamically loaded tool instances (e.g. from MCP servers).

        New tools are added to the full pool and automatically activated.
        """
        for t in tools:
            self._tool_map[t.name] = t
            if t.name not in self._all_tools:
                self._all_tools.append(t)
            if t.name.startswith("mcp__"):
                self.active_mcp_tools.add(t.name)
        self.bump_generation()

    def activate_mcp(self, names: Iterable[str]) -> tuple[list[str], list[str]]:
        """Activate MCP tools by name.

        Returns ``(added, unknown)`` — tools successfully activated and
        tools that were not found in the tool map or are not MCP tools.
        """
        added, unknown = [], []
        for name in names:
            if name not in self._tool_map or not name.startswith("mcp__"):
                unknown.append(name)
                continue
            if name not in self.active_mcp_tools:
                self.active_mcp_tools.add(name)
                added.append(name)

        if added:
            self._include = (self._include or set()) | set(added)
            self.bump_generation()

        return added, unknown

    def is_destructive(self, name: str, args: dict) -> bool:
        """Return ``True`` if a tool invocation may modify state."""
        if name == "shell":
            return args.get("action") in {"run", "start", "kill"}
        return name in DESTRUCTIVE_TOOLS or name.startswith("mcp__")

    def is_read_only(self, name: str, args: dict) -> bool:
        """Return ``True`` if a tool invocation is read-only."""
        if name == "shell":
            return args.get("action") in {"jobs", "read"}
        return name in READ_ONLY_TOOLS

    def _dedupe(self, tools):
        seen, out = set(), []
        for t in tools:
            if t.name and t.name not in seen: seen.add(t.name)
            out.append(t)
        return out

def coding_tools(*, include: list[str] | None = None, mcp: bool = False) -> ToolRegistry:
    """Convenience factory for selecting a subset of SDK tools by name.

    Example::

        agent = NessAgent(
            model=...,
            tools=coding_tools(include=["read", "grep", "glob"]),
            prompt=...,
        )

    Args:
        include: Tool names to include. When ``None``, all SDK tools are active.
        mcp: Reserved for future MCP integration.
    """
    return ToolRegistry(LOCAL_TOOLS, include=include)