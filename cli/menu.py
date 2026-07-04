"""Command catalog metadata.

The catalog is the single source of truth for which slash commands are exposed
by help, completion, and the full-screen input chrome.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CommandSpec:
    name: str
    summary: str
    group: str
    usage: str = ""


# Order matters for /help grouping and slash menu display.
COMMAND_CATALOG: tuple[CommandSpec, ...] = (
    CommandSpec("help", "Show the command reference", "General", "/help"),
    CommandSpec("config", "Set API keys, switch model, toggle options", "General", "/config"),
    CommandSpec("status", "Show session status and cache stats", "Session", "/status"),
    CommandSpec("skill", "List or load skill instructions", "Context", "/skill [<name>]"),
    CommandSpec("memory", "Read or append project memory", "Context", "/memory [add <note>]"),
    CommandSpec("user", "Read or append user preferences", "Context", "/user [add <note>]"),
    CommandSpec("threads", "List saved sessions", "Session", "/threads"),
    CommandSpec("resume", "Resume a saved thread", "Session", "/resume <id>"),
    CommandSpec("save", "Archive the current thread", "Session", "/save"),
    CommandSpec("reset", "Archive and start a fresh thread", "Session", "/reset"),
    CommandSpec("compact", "Force compaction on the next turn", "Session", "/compact"),
    CommandSpec("rollback", "Roll the thread back to a prior user turn", "Session", "/rollback [<seq>]"),
    CommandSpec("init", "Generate .ness/NESS.md", "Context", "/init [force]"),
    CommandSpec("permissions", "View or edit permission rules", "Tools", "/permissions"),
    CommandSpec("hooks", "List configured hooks", "Tools", "/hooks"),
    CommandSpec("mcp", "Show MCP server and tool status", "Tools", "/mcp"),
    CommandSpec("copy", "Copy assistant output", "Input", "/copy [code|<n>]"),
    CommandSpec("exit", "End the session", "General", "/exit"),
)

COMMAND_NAMES: tuple[str, ...] = tuple(spec.name for spec in COMMAND_CATALOG)
