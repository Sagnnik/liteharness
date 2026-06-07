from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import MemorySaver
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from agent import build_graph
from config import cost_tracker, settings
from hooks import describe_hooks
from mcp_client import mcp_manager
from memory import (
    MEMORY_FILE,
    USER_FILE,
    append_memory,
    append_user_memory,
    load_memory,
    load_user_memory,
    write_memory,
)
from permissions import list_rules, persist_rule, remove_rule
from context import build_init_memory_prompt
from session import (
    append_event,
    archive_thread,
    list_threads,
    load_thread_events,
)
from skill_loader import load_skill_errors, load_skills
from tools import is_git_repo, register_dynamic_tools
from tools.todo import get_thread_todos
from utils import get_project_context

console = Console()
assistant_history: list[str] = []
seen_counts: dict[str, int] = {}
bootstrap_messages: dict[str, list] = {}


async def run_turn(app, user_message: HumanMessage, thread_id: str, app_state: dict | None = None) -> None:
    """Run one graph turn and render streamed assistant/tool output."""
    state = app_state or {}
    agent_mode = state.get("agent_mode", "normal")
    append_event(thread_id, {"kind": "user", "content": _event_content(user_message.content)})
    config = {"configurable": {"thread_id": thread_id}}
    initial = bootstrap_messages.pop(thread_id, [])
    payload = {
        "messages": [*initial, user_message],
        "approval_declined": False,
        "agent_mode": agent_mode,
        "force_compact": bool(state.pop("force_compact", False)),
    }

    streaming_text = False
    async for event in app.astream_events(payload, config=config, version="v2"):
        if event.get("event") != "on_chat_model_stream":
            continue
        chunk = event.get("data", {}).get("chunk")
        text = getattr(chunk, "content", "")
        if not text or not isinstance(text, str):
            continue
        if not streaming_text:
            console.print()
            console.print(Text("Assistant: ", style="bold green"), end="")
            streaming_text = True
        console.print(text, end="")

    if streaming_text:
        console.print()

    await render_new_messages(app, config, thread_id, suppress_ai_text=streaming_text, agent_mode=agent_mode)
    render_todos(thread_id)


async def render_new_messages(
    app,
    config: dict,
    thread_id: str,
    suppress_ai_text: bool = False,
    agent_mode: str = "normal",
) -> None:
    try:
        snapshot = await app.aget_state(config)
        messages = list(snapshot.values.get("messages", []))
    except Exception:
        return

    seen = seen_counts.get(thread_id, 0)
    new_messages = messages[seen:]
    seen_counts[thread_id] = len(messages)

    for msg in new_messages:
        if msg.type in {"ai", "assistant"}:
            tool_calls = getattr(msg, "tool_calls", None) or []
            if tool_calls:
                for call in tool_calls:
                    render_tool_call(call)
            elif msg.content and not suppress_ai_text:
                text = str(msg.content)
                assistant_history.append(text)
                if agent_mode == "plan":
                    save_plan_text(thread_id, text)
                console.print()
                console.print(Panel(text, title="Assistant", border_style="green"))
            elif msg.content:
                text = str(msg.content)
                assistant_history.append(text)
                if agent_mode == "plan":
                    save_plan_text(thread_id, text)
        elif msg.type == "tool":
            preview = str(msg.content).replace("\n", " ")[:260]
            console.print(f"[dim]tool result:[/dim] {preview}")


def render_tool_call(call: dict[str, Any]) -> None:
    name = call.get("name", "unknown")
    args = call.get("args", {})
    arg_text = json.dumps(args, ensure_ascii=False)[:220] if isinstance(args, dict) else str(args)[:220]
    console.print(f"[yellow]tool:[/yellow] [cyan]{name}[/cyan] [dim]{arg_text}[/dim]")


def render_todos(thread_id: str) -> None:
    todos = get_thread_todos(thread_id)
    if not todos:
        return
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold")
    table.add_column("Status", style="cyan")
    table.add_column("ID", style="dim")
    table.add_column("Task")
    for todo in todos:
        table.add_row(todo["status"], todo["id"], todo["content"])
    console.print(Panel(table, title="Todos", border_style="blue"))


async def handle_command(command_line: str, model: ChatOpenRouter, app_state: dict) -> bool:
    """Handle slash commands. Return True when the command consumed the input."""
    raw = command_line[1:].strip()
    if not raw:
        return True
    name, _, args = raw.partition(" ")
    name = name.lower()

    if name in {"exit", "quit"}:
        app_state["exit"] = True
        return True
    if name == "reset":
        archive_thread(app_state["thread_id"])
        app_state["thread_id"] = new_thread_id()
        seen_counts.pop(app_state["thread_id"], None)
        console.print("[yellow]History cleared.[/yellow]")
        return True
    if name == "plan":
        app_state["agent_mode"] = "plan"
        app_state["rebuild_graph"] = True
        if args.strip():
            app_state["queued_prompt"] = args.strip()
        console.print("[cyan]Plan mode enabled. Tool set is read-only.[/cyan]")
        return True
    if name == "act":
        app_state["agent_mode"] = "normal"
        app_state["rebuild_graph"] = True
        if args.strip():
            app_state["queued_prompt"] = args.strip()
        console.print("[cyan]Normal mode enabled. Full tool set restored.[/cyan]")
        return True
    if name == "mode":
        console.print(f"Mode: {app_state.get('agent_mode', 'normal')}")
        return True
    if name == "context":
        console.print(Panel(Syntax(get_project_context(), "text"), title="Project Context", border_style="green"))
        return True
    if name == "cost":
        console.print(Panel(cost_tracker.report(), title="Session Cost", border_style="yellow"))
        return True
    if name == "cache":
        command_cache(app_state)
        return True
    if name == "skills":
        render_skills()
        return True
    if name == "init":
        await command_init(model, overwrite=args.strip() in {"force", "--force"})
        return True
    if name == "memory":
        command_memory(args)
        return True
    if name == "permissions":
        command_permissions(args)
        return True
    if name == "hooks":
        console.print(Panel(describe_hooks(), title="Hooks", border_style="cyan"))
        return True
    if name == "mcp":
        console.print(Panel(mcp_manager.status(), title="MCP", border_style="cyan"))
        return True
    if name == "threads":
        render_threads()
        return True
    if name == "resume":
        target = args.strip()
        if target and target != app_state["thread_id"]:
            archive_thread(app_state["thread_id"])
        command_resume(target, app_state)
        return True
    if name == "save":
        result = archive_thread(app_state["thread_id"])
        console.print(Panel(result, title="Save", border_style="green"))
        return True
    if name == "user":
        command_user(args)
        return True
    if name == "compact":
        app_state["force_compact"] = True
        append_event(app_state["thread_id"], {"kind": "compact", "content": "manual compaction requested"})
        console.print("[yellow]Compaction will run on the next model turn.[/yellow]")
        return True
    if name == "copy":
        command_copy(args.strip())
        return True
    if name == "image":
        app_state["pending_image"] = args.strip()
        console.print("[cyan]Image attached. Enter the question for this image next.[/cyan]")
        return True
    if name == "save-threads":
        command_save_threads(args.strip())
        return True

    disk_command = load_disk_commands().get(name)
    if disk_command:
        app_state["queued_prompt"] = disk_command.replace("{{args}}", args)
        return True

    console.print(f"[red]Unknown command:[/red] {name}")
    return True


async def command_init(model: ChatOpenRouter, overwrite: bool = False) -> None:
    context = get_project_context()
    response = await model.ainvoke([HumanMessage(content=build_init_memory_prompt(context))])
    result = write_memory(str(response.content), overwrite=overwrite)
    console.print(Panel(result, title="Init", border_style="green" if not result.startswith("Error:") else "red"))


def command_memory(args: str) -> None:
    args = args.strip()
    if not args:
        console.print(Panel(load_memory() or "(empty)", title=str(MEMORY_FILE), border_style="green"))
        return
    if args.startswith("add "):
        console.print(append_memory(args[4:]))
        return
    console.print("[red]Usage:[/red] /memory or /memory add <note>")


def command_user(args: str) -> None:
    args = args.strip()
    if not args:
        console.print(Panel(load_user_memory() or "(empty)", title=str(USER_FILE), border_style="green"))
        return
    if args.startswith("add "):
        console.print(append_user_memory(args[4:]))
        return
    console.print("[red]Usage:[/red] /user or /user add <preference>")


def command_permissions(args: str) -> None:
    parts = args.split()
    if not parts or parts[0] == "list":
        console.print(Panel(Syntax(list_rules(), "json"), title="Permissions", border_style="yellow"))
        return
    if len(parts) >= 2 and parts[0] in {"allow", "deny"}:
        persist_rule(" ".join(parts[1:]), parts[0])
        console.print(f"Added {parts[0]} rule.")
        return
    if len(parts) == 3 and parts[0] == "remove" and parts[1] in {"allow", "deny"}:
        removed = remove_rule(parts[1], int(parts[2]))
        console.print(f"Removed {removed}")
        return
    console.print("[red]Usage:[/red] /permissions [list|allow <pattern>|deny <pattern>|remove <allow|deny> <index>]")


def command_resume(thread_id: str, app_state: dict) -> None:
    if not thread_id:
        render_threads()
        return
    events = load_thread_events(thread_id)
    if not events:
        console.print(f"[red]No saved thread:[/red] {thread_id}")
        return
    bootstrap_messages[thread_id] = events_to_messages(events)
    app_state["thread_id"] = thread_id
    seen_counts[thread_id] = len(bootstrap_messages[thread_id])
    console.print(f"[green]Resumed thread:[/green] {thread_id}")


def command_copy(args: str) -> None:
    if not assistant_history:
        console.print("[yellow]No assistant message to copy.[/yellow]")
        return
    text = assistant_history[-1]
    if args == "code":
        blocks = re.findall(r"```(?:\w+)?\n(.*?)```", text, re.DOTALL)
        if blocks:
            text = blocks[-1]
    elif args.isdigit():
        idx = int(args)
        if 1 <= idx <= len(assistant_history):
            text = assistant_history[-idx]
    try:
        import pyperclip

        pyperclip.copy(text)
        console.print("[green]Copied.[/green]")
    except Exception:
        console.print(Panel(text, title="Clipboard unavailable", border_style="yellow"))


def command_save_threads(args: str) -> None:
    value = args.strip().lower()
    if value == "on":
        settings.auto_save_threads = True
    elif value == "off":
        settings.auto_save_threads = False
    console.print(f"Thread autosave: {'on' if settings.auto_save_threads else 'off'}")


def save_plan_text(thread_id: str, text: str) -> Path:
    plans_dir = Path(settings.ness_dir) / "plans"
    plans_dir.mkdir(parents=True, exist_ok=True)
    stamp = re.sub(r"[-:.TZ]", "", datetime.utcnow().isoformat(timespec="seconds"))
    path = plans_dir / f"{stamp}-{thread_id}.md"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    return path


def command_cache(app_state: dict) -> None:
    session_id = effective_openrouter_session_id(app_state["thread_id"])
    console.print(Panel(cost_tracker.report(session_id), title="Prompt Cache", border_style="cyan"))


def render_skills() -> None:
    skills = load_skills()
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Skill", style="cyan")
    table.add_column("Format")
    table.add_column("Source")
    table.add_column("Triggers")
    for skill in skills.values():
        table.add_row(
            skill.get("name", ""),
            skill.get("format", ""),
            skill.get("source", ""),
            ", ".join(skill.get("triggers", [])),
        )
    console.print(Panel(table, title="Skills", border_style="magenta"))
    errors = load_skill_errors()
    if errors:
        console.print(Panel("\n".join(errors), title="Skill Load Warnings", border_style="yellow"))


def render_threads() -> None:
    threads = list_threads(20)
    if not threads:
        console.print("[dim]No saved threads.[/dim]")
        return
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Thread")
    table.add_column("Summary")
    table.add_column("Turns", justify="right")
    table.add_column("Updated")
    table.add_column("Cost", justify="right")
    table.add_column("Cache hit", justify="right")
    for item in threads:
        input_tokens = int(item.get("input_tokens", 0) or 0)
        cached = int(item.get("cached_input_tokens", 0) or 0)
        cache_hit = cached / input_tokens if input_tokens else 0.0
        table.add_row(
            item.get("thread_id", ""),
            item.get("summary", "") or "(active)",
            str(item.get("turn_count", 0)),
            item.get("updated_at", ""),
            f"${float(item.get('total_cost_usd', 0.0)):.4f}",
            f"{cache_hit:.1%}",
        )
    console.print(Panel(table, title="Threads", border_style="cyan"))


def events_to_messages(events: list[dict]) -> list:
    messages = []
    for event in events:
        if event.get("kind") == "user":
            messages.append(HumanMessage(content=event.get("content", "")))
        elif event.get("kind") == "assistant" and event.get("content"):
            messages.append(AIMessage(content=event.get("content", "")))
    return messages[-20:]


def load_disk_commands() -> dict[str, str]:
    commands_dir = Path(settings.ness_dir) / "commands"
    if not commands_dir.exists():
        return {}
    commands = {}
    for path in commands_dir.glob("*.md"):
        text = path.read_text(encoding="utf-8")
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                yaml.safe_load(parts[1]) or {}
                text = parts[2]
        commands[path.stem] = text.strip()
    return commands


def build_user_message(text: str, pending_image: str = "") -> HumanMessage:
    text, inline_images = extract_inline_images(text)
    images = [pending_image] if pending_image else []
    images.extend(inline_images)
    if not images:
        return HumanMessage(content=text)
    if not settings.supports_vision:
        console.print("[yellow]Current model is not marked vision-capable; sending text only.[/yellow]")
        return HumanMessage(content=text)
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text or "Please inspect this image."}]
    for image_path in images:
        blocks.append({"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}})
    return HumanMessage(content=blocks)


def extract_inline_images(text: str) -> tuple[str, list[str]]:
    pattern = r"@image:([^\s]+)"
    images = re.findall(pattern, text)
    cleaned = re.sub(pattern, "", text).strip()
    return cleaned, images


def image_to_data_url(path: str) -> str:
    p = Path(path).expanduser()
    data = base64.b64encode(p.read_bytes()).decode("ascii")
    mime = mimetypes.guess_type(str(p))[0] or "image/png"
    return f"data:{mime};base64,{data}"


def _event_content(content) -> Any:
    if isinstance(content, (str, int, float, bool)) or content is None:
        return content
    return str(content)


def new_thread_id() -> str:
    return f"session-{uuid.uuid4().hex[:8]}"


def effective_openrouter_session_id(thread_id: str) -> str:
    return settings.openrouter_session_id or thread_id


def create_model(thread_id: str) -> ChatOpenRouter:
    model_kwargs = {
        "model": settings.model_name,
        "api_key": settings.openai_api_key,
        "session_id": effective_openrouter_session_id(thread_id),
    }
    if settings.openai_base_url:
        model_kwargs["base_url"] = settings.openai_base_url
    return ChatOpenRouter(**model_kwargs)


def render_header() -> None:
    table = Table(box=box.ROUNDED, show_header=False, padding=(0, 2))
    table.add_column(style="bold cyan", justify="center")
    table.add_row("LiteHarness")
    table.add_row(f"Mode: {settings.mode} | Model: {settings.model_name}")
    table.add_row(f"Approval: {'on' if settings.enable_approval else 'off'} | Autosave: {'on' if settings.auto_save_threads else 'off'}")
    console.print(Panel(table, border_style="bright_blue", title="Agent"))


async def main() -> None:
    render_header()
    await mcp_manager.start()
    register_dynamic_tools(mcp_manager.tools.values())

    git_available = is_git_repo()
    checkpointer = MemorySaver()
    app_state = {
        "thread_id": new_thread_id(),
        "exit": False,
        "pending_image": "",
        "queued_prompt": "",
        "agent_mode": "normal",
    }
    app_thread_id = app_state["thread_id"]
    model = create_model(app_thread_id)
    app = build_graph(
        model,
        thread_id=app_thread_id,
        agent_mode=app_state["agent_mode"],
        git_available=git_available,
        checkpointer=checkpointer,
    )

    try:
        while not app_state["exit"]:
            queued = app_state.pop("queued_prompt", "")
            if queued:
                user_input = queued
            else:
                try:
                    user_input = Prompt.ask("[bold cyan]You[/bold cyan]").strip()
                except (EOFError, KeyboardInterrupt):
                    break

            if not user_input:
                continue
            if user_input.startswith("/"):
                await handle_command(user_input, model, app_state)
                if app_state["thread_id"] != app_thread_id or app_state.pop("rebuild_graph", False):
                    app_thread_id = app_state["thread_id"]
                    model = create_model(app_thread_id)
                    app = build_graph(
                        model,
                        thread_id=app_thread_id,
                        agent_mode=app_state.get("agent_mode", "normal"),
                        git_available=git_available,
                        checkpointer=checkpointer,
                    )
                continue

            pending_image = app_state.pop("pending_image", "")
            await run_turn(app, build_user_message(user_input, pending_image), app_state["thread_id"], app_state)
    finally:
        try:
            archive_thread(app_state["thread_id"])
        except Exception as exc:
            console.print(f"[yellow]Archive skipped:[/yellow] {exc}")
        await mcp_manager.stop()
        console.print(Panel(cost_tracker.report(), title="Session Summary", border_style="bright_blue"))


if __name__ == "__main__":
    asyncio.run(main())
