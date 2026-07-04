"""Slash-command registry and handlers for the kept command set.

Each handler takes the SessionApp controller and the raw argument string. The
    dispatcher also resolves project-local disk commands (.ness/commands/*.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import yaml
from langchain_core.messages import HumanMessage

from config import cost_tracker, settings
from context import build_init_memory_prompt
from hooks import describe_hooks
from mcp_client import mcp_manager
from memory import (
    NESS_FILE,
    USER_FILE,
    append_ness_memory,
    append_user_memory,
    load_ness_memory,
    load_user_memory,
    write_ness_memory,
)
from model import active_model_name, active_reasoning_effort, effective_openrouter_session_id
from permissions import list_rules, persist_rule, remove_rule
from session import list_threads, list_user_turns
from skill_loader import load_skill_errors, load_skills
from utils import get_project_context

from cli import render
from cli.config_panel import run_config
from cli.menu import COMMAND_CATALOG

if TYPE_CHECKING:
    from cli.session_app import SessionApp

CommandHandler = Callable[["SessionApp", str], Awaitable[None]]

# --- handlers ---------------------------------------------------------------
async def cmd_exit(app: "SessionApp", args: str) -> None:
    app.should_exit = True


async def cmd_help(app: "SessionApp", args: str) -> None:
    rows: list[list[str]] = []
    for spec in COMMAND_CATALOG:
        rows.append([spec.usage or f"/{spec.name}", spec.summary])
    render.render_table(title="commands", columns=["command", "description"], rows=rows)
    render.render_notice("Shift+Tab toggles plan/act mode.")


async def cmd_config(app: "SessionApp", args: str) -> None:
    result = await run_config()
    for message in result.messages:
        render.render_notice(message)
    if result.rebuild:
        app.rebuild_graph()
    if result.session_update:
        app.render_header()


async def cmd_status(app: "SessionApp", args: str) -> None:
    session_id = effective_openrouter_session_id(app.thread_id)
    input_tokens = int(cost_tracker.input_tokens or 0)
    cached = int(cost_tracker.cached_input_tokens or 0)
    cache_hit = cached / input_tokens if input_tokens else None
    lines = [
        f"session id     {app.thread_id}",
        f"model          {active_model_name()}",
        f"reasoning      {active_reasoning_effort()}",
        f"input tokens   {input_tokens:,}",
        f"output tokens  {int(cost_tracker.output_tokens or 0):,}",
        f"cost           ${cost_tracker.cost_usd:.4f}" if cost_tracker.cost_usd > 0 else "cost           unknown",
        f"turns          {int(getattr(app, 'turn_count', 0) or 0)}",
        f"cache read     {cached:,}",
        f"cache hit      {cache_hit:.0%}" if cache_hit is not None else "cache hit      n/a",
        f"openrouter id  {session_id or 'not set'}",
    ]
    render.render_panel_text("\n".join(lines), title="session status", style="usage.value")


async def cmd_skill(app: "SessionApp", args: str) -> None:
    name = args.strip()
    skills = load_skills()
    if not name:
        if not skills:
            render.render_notice("No skills found under .ness/skills/.")
        else:
            rows = [
                [s.get("name", ""), s.get("source", ""), ", ".join(s.get("triggers", []))]
                for s in skills.values()
            ]
            render.render_table(title="skills", columns=["skill", "source", "triggers"], rows=rows)
        errors = load_skill_errors()
        if errors:
            render.render_warning("Skill load warnings:\n" + "\n".join(errors))
        render.render_notice("Load a skill with /skill <name>.")
        return
    if name not in skills:
        render.render_error(f"Unknown skill: {name}  (/skill to list)")
        return
    if name not in app.pending_skills:
        app.pending_skills.append(name)
    render.render_notice(f"Skill '{name}' will load on your next message.", title="skill")


async def cmd_init(app: "SessionApp", args: str) -> None:
    force = args.strip() in {"force", "--force"}
    with render.thinking("generating NESS.md"):
        response = await app.model.ainvoke([HumanMessage(content=build_init_memory_prompt(get_project_context()))])
    result = write_ness_memory(str(response.content), overwrite=force)
    if result.startswith("Error:"):
        render.render_error(result)
    else:
        render.render_notice(result, title="init")


async def cmd_memory(app: "SessionApp", args: str) -> None:
    args = args.strip()
    if not args:
        render.render_panel_text(load_ness_memory() or "(empty)", title=str(NESS_FILE), style="usage.value")
        return
    if args.startswith("add "):
        render.render_notice(append_ness_memory(args[4:]))
        return
    render.render_error("Usage: /memory or /memory add <note>")


async def cmd_user(app: "SessionApp", args: str) -> None:
    args = args.strip()
    if not args:
        render.render_panel_text(load_user_memory() or "(empty)", title=str(USER_FILE), style="usage.value")
        return
    if args.startswith("add "):
        render.render_notice(append_user_memory(args[4:]))
        return
    render.render_error("Usage: /user or /user add <preference>")


async def cmd_permissions(app: "SessionApp", args: str) -> None:
    parts = args.split()
    if not parts or parts[0] == "list":
        render.render_panel_text(list_rules(), title="permissions", style="usage.value")
        return
    if len(parts) >= 2 and parts[0] in {"allow", "deny"}:
        persist_rule(" ".join(parts[1:]), parts[0])
        render.render_notice(f"Added {parts[0]} rule.")
        return
    if len(parts) == 3 and parts[0] == "remove" and parts[1] in {"allow", "deny"}:
        try:
            removed = remove_rule(parts[1], int(parts[2]))
            render.render_notice(f"Removed {removed}")
        except ValueError as exc:
            render.render_error(str(exc))
        return
    render.render_error("Usage: /permissions [list | allow <pattern> | deny <pattern> | remove <allow|deny> <index>]")


async def cmd_hooks(app: "SessionApp", args: str) -> None:
    render.render_panel_text(describe_hooks(), title="hooks", style="usage.value")


async def cmd_mcp(app: "SessionApp", args: str) -> None:
    parts = args.split()
    if not parts:
        render.render_panel_text(mcp_manager.status(), title="mcp", style="usage.value")
        return

    from tools import activate_mcp_tools, tool_names_for_session
    from tools.discover import TOOL_COUNT_WARN_THRESHOLD

    server = parts[0]
    catalog = mcp_manager.catalog()
    server_info = catalog.get(server)
    if server_info is None:
        render.render_error(f"Unknown MCP server: {server}  (/mcp for status)")
        return

    entries = server_info.get("tools", [])
    if len(parts) >= 2:
        tool_short = parts[1]
        wanted = [e["name"] for e in entries if e.get("tool") == tool_short]
        if not wanted:
            render.render_error(f"Unknown tool '{tool_short}' on server '{server}'.")
            return
    else:
        wanted = [e["name"] for e in entries]

    added, unknown = activate_mcp_tools(wanted)
    if added:
        render.render_notice(
            f"Loaded {len(added)} tool(s) from {server}: {', '.join(sorted(added))}",
            title="mcp",
        )
    else:
        render.render_notice(f"No new tools loaded from {server}.", title="mcp")
    if unknown:
        render.render_warning(f"Skipped unknown: {', '.join(sorted(set(unknown)))}")

    total = len(tool_names_for_session())
    if total > TOOL_COUNT_WARN_THRESHOLD:
        render.render_warning(
            f"{total} tools now loaded (> {TOOL_COUNT_WARN_THRESHOLD}); "
            "tool-selection accuracy may degrade."
        )


async def cmd_threads(app: "SessionApp", args: str) -> None:
    threads = list_threads(20)
    if not threads:
        render.render_notice("No saved sessions.")
        return
    rows = []
    for item in threads:
        input_tokens = int(item.get("input_tokens", 0) or 0)
        cached = int(item.get("cached_input_tokens", 0) or 0)
        cache_hit = cached / input_tokens if input_tokens else 0.0
        rows.append(
            [
                item.get("thread_id", ""),
                item.get("summary", "") or "(active)",
                str(item.get("turn_count", 0)),
                f"${float(item.get('total_cost_usd', 0.0)):.4f}",
                f"{cache_hit:.0%}",
            ]
        )
    render.render_table(title="threads", columns=["thread", "summary", "turns", "cost", "cache"], rows=rows)


async def cmd_resume(app: "SessionApp", args: str) -> None:
    target = args.strip()
    if not target:
        await cmd_threads(app, "")
        render.render_notice("Usage: /resume <thread_id>")
        return
    await app.resume_thread(target)


async def cmd_save(app: "SessionApp", args: str) -> None:
    render.render_notice(app.save_thread(), title="save")


async def cmd_reset(app: "SessionApp", args: str) -> None:
    await app.reset_thread()
    render.render_notice("Started a fresh thread.")


async def cmd_compact(app: "SessionApp", args: str) -> None:
    app.request_compact()
    render.render_notice("Compaction will run on the next model turn.")


async def cmd_copy(app: "SessionApp", args: str) -> None:
    history = app.assistant_history
    if not history:
        render.render_warning("No assistant message to copy.")
        return
    text = history[-1]
    args = args.strip()
    if args == "code":
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[-1]
    elif args.isdigit():
        idx = int(args)
        if 1 <= idx <= len(history):
            text = history[-idx]
    try:
        import pyperclip

        pyperclip.copy(text)
        render.render_notice("Copied to clipboard.")
    except Exception:
        render.render_panel_text(text, title="clipboard unavailable", style="usage.value")


async def cmd_rollback(app: "SessionApp", args: str) -> None:
    """Roll the current thread back to a prior user turn.

    Usage:
      /rollback                open a picker of every user message in this thread
      /rollback <seq>          roll back directly to the user message at seq N
                               (use /status or the picker to find the seq)

    Restores agent-modified files (git stash create snapshot), the per-thread
    session memory file, and truncates the durable events tail at the chosen
    user message. The in-process cost_tracker is intentionally preserved.
    """
    arg = args.strip()
    if arg.isdigit():
        await app.rollback_to(int(arg))
        return

    sink = render.get_sink()
    if sink is None:
        render.render_error("/rollback picker requires the interactive TUI; use /rollback <seq>.")
        return

    turns = list_user_turns(app.thread_id)
    if not turns:
        render.render_notice("No user turns in this thread to roll back to.")
        return

    seq_str = await sink.request_rollback_picker(turns)
    if not seq_str:
        return  # user cancelled the picker
    try:
        seq = int(seq_str)
    except ValueError:
        render.render_error(f"Invalid rollback seq: {seq_str!r}")
        return
    await app.rollback_to(seq)


HANDLERS: dict[str, CommandHandler] = {
    "exit": cmd_exit,
    "quit": cmd_exit,
    "help": cmd_help,
    "config": cmd_config,
    "status": cmd_status,
    "skill": cmd_skill,
    "init": cmd_init,
    "memory": cmd_memory,
    "user": cmd_user,
    "permissions": cmd_permissions,
    "hooks": cmd_hooks,
    "mcp": cmd_mcp,
    "threads": cmd_threads,
    "resume": cmd_resume,
    "save": cmd_save,
    "reset": cmd_reset,
    "compact": cmd_compact,
    "copy": cmd_copy,
    "rollback": cmd_rollback,
}

# Slash commands safe to run while a task is streaming: read-only or file-write
# side effects that do not touch the live graph or thread state.
BUSY_SAFE_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "status",
        "threads",
        "permissions",
        "hooks",
        "mcp",
        "copy",
        "memory",
        "user",
        "skill",
    }
)


async def dispatch(app: "SessionApp", command_line: str, *, busy: bool = False) -> None:
    raw = command_line[1:].strip()
    if not raw:
        return
    name, _, args = raw.partition(" ")
    name = name.lower()

    handler = HANDLERS.get(name)
    if handler is not None:
        if busy and name not in BUSY_SAFE_COMMANDS:
            render.render_warning(f"/{name} is not available while a task is running")
            return
        await handler(app, args)
        return

    disk_command = _load_disk_commands().get(name)
    if disk_command is not None:
        app.queued_prompt = disk_command.replace("{{args}}", args)
        return

    if busy:
        render.render_warning(f"/{name} is not available while a task is running")
        return
    render.render_error(f"Unknown command: /{name}  (try /help)")


def _load_disk_commands() -> dict[str, str]:
    commands_dir = Path(settings.ness_dir) / "commands"
    if not commands_dir.exists():
        return {}
    commands: dict[str, str] = {}
    for path in commands_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                yaml.safe_load(parts[1]) or {}
                text = parts[2]
        commands[path.stem] = text.strip()
    return commands
