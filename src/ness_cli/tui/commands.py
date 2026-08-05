"""Slash-command registry and handlers for the kept command set.

Each handler takes the TuiApp and the raw argument string. Session-level
operations delegate to ``app.coding`` (the CodingSession) or the TuiApp
facade; the dispatcher also resolves project-local disk commands
(.ness/commands/*.md).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Awaitable, Callable

import yaml
from langchain_core.messages import HumanMessage

from ness_agent.tools.discover import TOOL_COUNT_WARN_THRESHOLD
from ness_agent.workspace import setup_ness_structure
from ness_agent.workspace.project_context import get_project_context
from ness_cli.chat_model import (
    active_model_name,
    active_reasoning_effort,
    openrouter_session,
)
from ness_cli.config import settings
from ness_cli.prompts import build_init_memory_prompt

from ness_cli.tui import render
from ness_cli.tui.config_flow import run_config
from ness_cli.tui.command_catalog import COMMAND_CATALOG

if TYPE_CHECKING:
    from ness_cli.tui.app import TuiApp

CommandHandler = Callable[["TuiApp", str], Awaitable[None]]


def _add_text(args: str) -> str | None:
    """Return the text after ``add `` if present, else ``None``."""
    args = args.strip()
    return args[4:].strip() if args.startswith("add ") else None


# --- handlers ---------------------------------------------------------------
async def cmd_exit(app: "TuiApp", args: str) -> None:
    app.should_exit = True


async def cmd_help(app: "TuiApp", args: str) -> None:
    rows: list[list[str]] = []
    for spec in COMMAND_CATALOG:
        rows.append([spec.usage or f"/{spec.name}", spec.summary])
    render.render_table(title="commands", columns=["command", "description"], rows=rows)
    render.render_notice("Shift+Tab toggles plan/act mode.")


async def cmd_config(app: "TuiApp", args: str) -> None:
    result = await run_config()
    for message in result.messages:
        render.render_notice(message)
    if result.rebuild:
        app.rebuild_graph()
        await app.refresh_context_snapshot()
    if result.session_update:
        options = app.coding.cfg.options
        options.enable_approval = (
            settings.enable_approval and not getattr(options, "yolo_mode", False)
        )
        options.auto_save_threads = settings.auto_save_threads
        options.session_end_reflection = settings.session_end_reflection
        app.coding.thread_store.auto_save = settings.auto_save_threads
        app.render_header()


async def cmd_status(app: "TuiApp", args: str) -> None:
    session_id = openrouter_session(app.thread_id)
    tracker = app.coding.cost_tracker
    input_tokens = int(tracker.input_tokens or 0)
    cached = int(tracker.cached_input_tokens or 0)
    cache_hit = cached / input_tokens if input_tokens else None
    lines = [
        f"session id     {app.thread_id}",
        f"model          {active_model_name()}",
        f"reasoning      {active_reasoning_effort()}",
        f"input tokens   {input_tokens:,}",
        f"output tokens  {int(tracker.output_tokens or 0):,}",
        f"cost           ${tracker.cost_usd:.4f}" if tracker.cost_usd > 0 else "cost           unknown",
        f"turns          {int(app.turn_count or 0)}",
        f"cache read     {cached:,}",
        f"cache hit      {cache_hit:.0%}" if cache_hit is not None else "cache hit      n/a",
        f"openrouter id  {session_id or 'not set'}",
    ]
    render.render_panel_text("\n".join(lines), title="session status", style="usage.value")


async def cmd_skill(app: "TuiApp", args: str) -> None:
    name = args.strip()
    loader = app.coding.skill_loader
    skills = loader.load()
    if not name:
        if not skills:
            render.render_notice("No skills found.")
        else:
            rows = [
                [s.get("name", ""), s.get("source", ""), s.get("description", "")]
                for s in skills.values()
            ]
            render.render_table(title="skills", columns=["skill", "source", "description"], rows=rows)
        if loader.errors:
            render.render_warning("Skill load warnings:\n" + "\n".join(loader.errors))
        render.render_notice("Load a skill with /skill <name>.")
        return
    if name not in skills:
        render.render_error(f"Unknown skill: {name}  (/skill to list)")
        return
    app.coding.stage_skills([name])
    render.render_notice(f"Skill '{name}' will load on your next message.", title="skill")


async def cmd_init(app: "TuiApp", args: str) -> None:
    from ness_cli.paths import ensure_global_config, resolve_paths

    paths = resolve_paths(
        project_root=app.coding.project_root,
        ness_dir=app.coding.ness_dir,
    )
    created = setup_ness_structure(app.coding.ness_dir)
    created.extend(ensure_global_config(paths))
    if created:
        render.render_notice(
            f"Initialized .ness/ + global config ({', '.join(created)})",
            title="init",
        )
    else:
        render.render_notice(".ness/ structure already present", title="init")


async def cmd_memory(app: "TuiApp", args: str) -> None:
    memory = app.coding.memory_store
    raw = args.strip()
    if raw.startswith("create"):
        rest = raw[6:].strip()
        force = rest in ("force", "--force")
        if rest and not force:
            render.render_error("Usage: /memory create [force]")
            return
        with render.thinking("generating NESS.md"):
            response = await app.model.ainvoke(
                [HumanMessage(content=build_init_memory_prompt(
                    get_project_context(),
                    instructions_dir=app.coding.instructions_dir,
                ))]
            )
        result = memory.write_project(str(response.content), overwrite=force)
        if result.startswith("Error:"):
            render.render_error(result)
        else:
            render.render_notice(result, title="memory")
        return

    text = _add_text(args)
    if text is None:
        if not raw:
            render.render_panel_text(memory.load_project() or "(empty)", title=str(memory.ness_file), style="usage.value")
            return
        render.render_error("Usage: /memory or /memory add <note> or /memory create [force]")
        return
    render.render_notice(memory.append_project(text))


async def cmd_user(app: "TuiApp", args: str) -> None:
    memory = app.coding.memory_store
    text = _add_text(args)
    if text is None:
        if not args.strip():
            render.render_panel_text(memory.load_user() or "(empty)", title=str(memory.user_file), style="usage.value")
            return
        render.render_error("Usage: /user or /user add <preference>")
        return
    render.render_notice(memory.append_user(text))


async def cmd_permissions(app: "TuiApp", args: str) -> None:
    permission_store = app.coding.permission_store
    parts = args.split()
    if not parts or parts[0] == "list":
        render.render_panel_text(permission_store.list_rules(), title="permissions", style="usage.value")
        return
    if len(parts) >= 2 and parts[0] in {"allow", "deny"}:
        permission_store.persist_rule(" ".join(parts[1:]), parts[0])
        render.render_notice(f"Added {parts[0]} rule.")
        return
    if len(parts) == 3 and parts[0] == "remove" and parts[1] in {"allow", "deny"}:
        try:
            removed = permission_store.remove_rule(parts[1], int(parts[2]))
            render.render_notice(f"Removed {removed}")
        except ValueError as exc:
            render.render_error(str(exc))
        return
    render.render_error("Usage: /permissions [list | allow <pattern> | deny <pattern> | remove <allow|deny> <index>]")


async def cmd_hooks(app: "TuiApp", args: str) -> None:
    render.render_panel_text(app.coding.hook_runner.describe(), title="hooks", style="usage.value")


async def cmd_mcp(app: "TuiApp", args: str) -> None:
    if app.mcp is None:
        render.render_error("/mcp is unavailable: no MCP manager configured.")
        return
    parts = args.split()
    if not parts:
        render.render_panel_text(app.mcp.status(), title="mcp", style="usage.value")
        return

    server = parts[0]
    catalog = app.mcp.catalog()
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

    added, unknown = app.coding.tool_registry.activate_mcp(wanted)
    if added:
        render.render_notice(
            f"Loaded {len(added)} tool(s) from {server}: {', '.join(sorted(added))}",
            title="mcp",
        )
    else:
        render.render_notice(f"No new tools loaded from {server}.", title="mcp")
    if unknown:
        render.render_warning(f"Skipped unknown: {', '.join(sorted(set(unknown)))}")

    total = len(app.coding.tool_registry.tool_names())
    if total > TOOL_COUNT_WARN_THRESHOLD:
        render.render_warning(
            f"{total} tools now loaded (> {TOOL_COUNT_WARN_THRESHOLD}); "
            "tool-selection accuracy may degrade."
        )


def _thread_rows(threads: list[dict], store) -> list[list[str]]:
    rows: list[list[str]] = []
    for item in threads:
        input_tokens = int(item.get("input_tokens", 0) or 0)
        cached = int(item.get("cached_input_tokens", 0) or 0)
        cache_hit = cached / input_tokens if input_tokens else 0.0
        label = item.get("summary") or store.first_user_message(item.get("thread_id", "")) or "(no messages)"
        if "archived_at" not in item:
            label = f"{label} (active)"
        rows.append(
            [
                item.get("thread_id", ""),
                label,
                str(item.get("turn_count", 0)),
                f"${float(item.get('total_cost_usd', 0.0)):.4f}",
                f"{cache_hit:.0%}",
            ]
        )
    return rows


async def cmd_threads(app: "TuiApp", args: str) -> None:
    store = app.coding.thread_store
    threads = store.list_threads(100)
    if not threads:
        render.render_notice("No saved sessions.")
        return
    sink = render.get_sink()
    if sink is None:
        render.render_error("/threads requires the interactive TUI.")
        return
    for item in threads:
        item["label"] = (
            item.get("summary")
            or store.first_user_message(item.get("thread_id", ""))
            or "(no messages)"
        )
    target = await sink.request_threads_picker(
        threads,
        current_thread_id=app.thread_id,
    )
    if target and target != app.thread_id:
        await app.resume_thread(target)


async def cmd_save(app: "TuiApp", args: str) -> None:
    render.render_notice(app.save_thread(), title="save")


async def cmd_new(app: "TuiApp", args: str) -> None:
    await app.reset_thread()
    render.render_notice("Started a fresh thread.")


async def cmd_compact(app: "TuiApp", args: str) -> None:
    app.request_compact()
    render.render_notice("Compaction will run on the next model turn.")


async def cmd_copy(app: "TuiApp", args: str) -> None:
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


async def cmd_rollback(app: "TuiApp", args: str) -> None:
    """Roll the current thread back to a prior user turn.

    Usage:
      /rollback                open a picker of every user message in this thread
      /rollback <seq>          roll back directly to the user message at seq N
                               (use /status or the picker to find the seq)

    Restores agent-modified files (git snapshot), the per-thread session
    memory file, and truncates the durable events tail at the chosen user
    message. The in-process cost tracker is intentionally preserved.
    """
    arg = args.strip()
    if arg.isdigit():
        await app.rollback_to(int(arg))
        return

    sink = render.get_sink()
    if sink is None:
        render.render_error("/rollback picker requires the interactive TUI; use /rollback <seq>.")
        return

    turns = app.coding.thread_store.list_user_turns(app.thread_id)
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


async def cmd_fork(app: "TuiApp", args: str) -> None:
    turns = app.coding.thread_store.list_user_turns(app.thread_id)
    if not turns:
        render.render_notice("No user turns in this thread to fork from.")
        return
    sink = render.get_sink()
    if sink is None:
        render.render_error("/fork requires the interactive TUI.")
        return
    seq_str = await sink.request_fork_picker(turns)
    if not seq_str:
        return
    try:
        await app.fork_thread(int(seq_str))
    except ValueError as exc:
        render.render_error(str(exc))


async def cmd_goal(app: "TuiApp", args: str) -> None:
    goal = args.strip()
    if not goal:
        render.render_error("Usage: /goal <objective>")
        return
    await app.run_goal(goal)


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
    "fork": cmd_fork,
    "goal": cmd_goal,
    "save": cmd_save,
    "new": cmd_new,
    "compact": cmd_compact,
    "copy": cmd_copy,
    "rollback": cmd_rollback,
}

# Slash commands safe to run while a task is streaming: read-only or file-write
# side effects that do not touch the live graph or thread state.
# Exception: /memory create invokes the chat model and is refused when busy
# (see ``dispatch``); /memory read and /memory add remain allowed.
BUSY_SAFE_COMMANDS: frozenset[str] = frozenset(
    {
        "help",
        "status",
        "permissions",
        "hooks",
        "mcp",
        "copy",
        "memory",
        "user",
        "skill",
    }
)


async def dispatch(app: "TuiApp", command_line: str, *, busy: bool = False) -> None:
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
        if busy and name == "memory" and args.strip().startswith("create"):
            render.render_warning("/memory create is not available while a task is running")
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
