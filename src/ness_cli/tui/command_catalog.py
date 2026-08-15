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
    CommandSpec("login", "Connect or switch a model provider", "General", "/login"),
    CommandSpec("config", "Edit provider, model, and behavior settings", "General", "/config"),
    CommandSpec("status", "Show provider, account, limits, and session usage", "Session", "/status"),
    CommandSpec("skill", "List or load skill instructions", "Context", "/skill [<name>]"),
    CommandSpec("memory", "Read, append, or draft project memory", "Context", "/memory [add <note>|create [force]]"),
    CommandSpec("user", "Read or append user preferences", "Context", "/user [add <note>]"),
    CommandSpec("threads", "Select and switch saved sessions", "Session", "/threads"),
    CommandSpec("rename", "Set the current session name", "Session", "/rename <name>"),
    CommandSpec("fork", "Fork before a prior user message", "Session", "/fork"),
    CommandSpec("goal", "Run a bounded worker–judge objective loop", "Session", "/goal <objective>"),
    CommandSpec("save", "Archive the current thread", "Session", "/save"),
    CommandSpec("new", "Archive and start a fresh thread", "Session", "/new"),
    CommandSpec("compact", "Force compaction on the next turn", "Session", "/compact"),
    CommandSpec("reflection", "Reflect on new conversation history now", "Session", "/reflection"),
    CommandSpec("export", "Export the full session to HTML", "Session", "/export <path.html>"),
    CommandSpec("rollback", "Roll the thread back to a prior user turn", "Session", "/rollback [<seq>]"),
    CommandSpec("init", "Initialize .ness/ (dirs, defaults, empty NESS.md)", "Context", "/init"),
    CommandSpec("permissions", "View or edit permission rules", "Tools", "/permissions"),
    CommandSpec("hooks", "List configured hooks", "Tools", "/hooks"),
    CommandSpec("mcp", "Show MCP server and tool status", "Tools", "/mcp"),
    CommandSpec("clear", "Clear the transcript display", "Input", "/clear"),
    CommandSpec("copy", "Copy assistant output", "Input", "/copy [code|<n>]"),
    CommandSpec("exit", "End the session", "General", "/exit"),
)
