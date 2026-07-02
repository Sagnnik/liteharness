"""Selective tool-call summaries for the CLI transcript."""

from __future__ import annotations

from pathlib import Path
from typing import Any

# Tool results omitted from the transcript (output still goes to the model).
SILENT_RESULT_TOOLS = frozenset(
    {
        "read_file",
        "grep",
        "glob_files",
        "list_files",
        "check_syntax",
        "todo",
    }
)

# Consecutive calls with the same name are merged onto one line.
BATCHABLE_TOOL_CALLS = frozenset(SILENT_RESULT_TOOLS)

_PROMPT_PREVIEW = 160
_DEFAULT_ARG_PREVIEW = 120


def should_show_tool_result(name: str) -> bool:
    return name not in SILENT_RESULT_TOOLS


def should_show_tool_call(name: str) -> bool:
    return True


def _truncate(text: str, limit: int) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 1)] + "…"


def _short_path(path: str) -> str:
    raw = str(path or "").strip()
    if not raw:
        return "?"
    try:
        resolved = Path(raw).expanduser()
        return str(resolved.relative_to(Path.cwd()))
    except ValueError:
        return raw


def _read_file_token(args: dict[str, Any]) -> str:
    path = _short_path(str(args.get("path", "")))
    offset = int(args.get("offset", 1) or 1)
    if offset > 1:
        return f"{offset}| {path}"
    return path


def _grep_token(args: dict[str, Any]) -> str:
    pattern = _truncate(str(args.get("pattern", "")), 40)
    path = _short_path(str(args.get("path", ".")))
    if path and path != ".":
        return f"{pattern}  {path}"
    return pattern


def _glob_token(args: dict[str, Any]) -> str:
    pattern = str(args.get("pattern", "") or args.get("glob", "") or "?")
    path = _short_path(str(args.get("path", ".")))
    if path and path != ".":
        return f"{pattern}  {path}"
    return pattern


def _list_files_token(args: dict[str, Any]) -> str:
    return _short_path(str(args.get("path", ".")))


def spawn_subagent_task_rows(args: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (agent_name, prompt) rows for spawn_subagent calls."""
    return _spawn_subagent_lines(args)


def _spawn_subagent_lines(args: dict[str, Any]) -> list[tuple[str, str]]:
    """Return (name, args_text) rows; args_text may contain newlines."""
    tasks = args.get("tasks")
    if isinstance(tasks, list) and tasks:
        rows: list[tuple[str, str]] = []
        for task in tasks:
            if not isinstance(task, dict):
                continue
            agent = str(task.get("name", "?"))
            prompt = _truncate(str(task.get("prompt", "")), _PROMPT_PREVIEW)
            rows.append((agent, prompt))
        if rows:
            return rows

    agent = str(args.get("name", "?"))
    prompt = _truncate(str(args.get("prompt", "")), _PROMPT_PREVIEW)
    return [(agent, prompt)]


def _generic_args_text(args: Any) -> str:
    if not isinstance(args, dict) or not args:
        return ""
    parts: list[str] = []
    for key, value in args.items():
        if value in (None, "", [], {}):
            continue
        parts.append(f"{key}={_truncate(value, 48)}")
    return _truncate("  ".join(parts), _DEFAULT_ARG_PREVIEW)


def format_tool_args(name: str, args: Any) -> str:
    """Single-line args summary for unknown / generic tools."""
    if not isinstance(args, dict):
        return _truncate(str(args), _DEFAULT_ARG_PREVIEW)

    if name == "read_file":
        return _read_file_token(args)
    if name == "grep":
        return _grep_token(args)
    if name == "glob_files":
        return _glob_token(args)
    if name == "list_files":
        return _list_files_token(args)
    if name == "spawn_subagent":
        rows = _spawn_subagent_lines(args)
        if len(rows) == 1:
            agent, prompt = rows[0]
            return f"{agent}  {prompt}".strip() if prompt else agent
        return "\n".join(f"{agent}  {prompt}".strip() for agent, prompt in rows)
    if name == "todo":
        return ""
    if name == "shell":
        cmd = _truncate(str(args.get("command", "")), _DEFAULT_ARG_PREVIEW)
        return cmd
    if name in {"write_file", "edit", "delete_file"}:
        path = _short_path(str(args.get("path", "")))
        return path
    if name == "web_search":
        return _truncate(str(args.get("query", "")), _DEFAULT_ARG_PREVIEW)
    if name == "fetch_url":
        return _truncate(str(args.get("url", "")), _DEFAULT_ARG_PREVIEW)
    return _generic_args_text(args)


def format_batched_tool_args(name: str, calls: list[dict[str, Any]]) -> str:
    """Side-by-side summary for a run of identical batchable tool calls."""
    tokens: list[str] = []
    for call in calls:
        args = call.get("args") or {}
        if not isinstance(args, dict):
            continue
        if name == "read_file":
            tokens.append(_read_file_token(args))
        elif name == "grep":
            tokens.append(_grep_token(args))
        elif name == "glob_files":
            tokens.append(_glob_token(args))
        elif name == "list_files":
            tokens.append(_list_files_token(args))
        else:
            tokens.append(format_tool_args(name, args))
    return "  ".join(token for token in tokens if token)


def spawn_subagent_result_summary(content: str) -> str:
    """One-line summary for a spawn_subagent tool result."""
    text = str(content or "").strip()
    if not text:
        return "subagent finished"
    first = text.splitlines()[0]
    if first.startswith("status="):
        status = first.split("=", 1)[-1]
        return f"subagent batch {status}"
    if text.startswith("Error:") or "Error:" in text[:80]:
        return _truncate(text.replace("\n", " "), 100)
    return "subagent finished"

