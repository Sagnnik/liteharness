"""Preview prompt context layers for plan or act mode.

Usage:
  PYTHONPATH=. uv run python tests/preview_context.py
  PYTHONPATH=. uv run python tests/preview_context.py --mode plan --query "Add retry logic to web tools"
  PYTHONPATH=. uv run python tests/preview_context.py --mode both --tokens 90000
  PYTHONPATH=. uv run python tests/preview_context.py --diff-modes
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from langchain_core.messages import SystemMessage
from langchain_core.utils.function_calling import convert_to_openai_tool

from compaction import (
    CompactionResult,
    format_compaction_overlay_note,
    resolve_token_count,
    resolve_usable_context_budget,
)
from config import settings
from context import (
    DEFAULT_PERSONA,
    build_l0,
    build_l1,
    build_project_context_block,
    build_working_state_sections,
    render_todos,
)
from memory import load_ness_memory, load_repo_context, load_user_memory
from skill_loader import load_skills, render_skill_catalog, select_sticky_skills
from tools import is_git_repo, select_tools_for_session
from git_context import git_worktree_summary

app = typer.Typer(add_completion=False, help="Preview LiteHarness L1/L2/L3 context layers.")
console = Console()


def _resolve_git(choice: str) -> bool:
    if choice == "auto":
        return is_git_repo()
    return choice == "yes"


def _sample_todos(mode: str) -> list[dict]:
    if mode == "plan":
        return []
    return [
        {"id": "1", "content": "Read tools/web.py and tests/test_web.py", "status": "completed"},
        {"id": "2", "content": "Add retry wrapper to webfetch", "status": "in_progress"},
        {"id": "3", "content": "Add tests for retry behavior", "status": "pending"},
    ]


def _working_state_tail(overlay: str) -> str:
    return f"<system-reminder>\n{overlay.strip()}\n</system-reminder>"


def _tool_api_metadata(tools: list) -> list[dict]:
    """OpenAI-style tool definitions from LangChain (same as bind_tools().kwargs['tools'])."""
    return [convert_to_openai_tool(tool) for tool in tools]


def _render_tools_metadata(tools: list) -> str:
    return json.dumps(_tool_api_metadata(tools), indent=2, sort_keys=True, default=str)


def _tokens(text: str) -> int:
    if not text.strip():
        return 0
    return resolve_token_count([SystemMessage(content=text)], known_input_tokens=None)


STATIC_PREFIX_TOKEN_TARGET = 7000


def _panel(title: str, body: str, *, style: str = "cyan") -> Panel:
    return Panel(
        Text(body.strip() or "(empty)", overflow="fold"),
        title=title,
        border_style=style,
        box=box.ROUNDED,
        padding=(1, 2),
    )


def _build_mode_parts(
    mode: str,
    *,
    query: str,
    git_available: bool,
    token_count: int | None,
    user_message_count: int,
):
    tools = select_tools_for_session()
    tool_names = {tool.name for tool in tools}

    all_skills = load_skills()
    sticky: set[str] = set()
    active_skills = select_sticky_skills(query, all_skills, sticky)
    skill_catalog = render_skill_catalog(all_skills)

    l1 = "\n\n".join(
        [
            build_l0(tools),
            build_l1(DEFAULT_PERSONA, tools, load_user_memory(), load_ness_memory(), skill_catalog),
        ]
    ).strip()
    l2 = build_project_context_block(
        load_repo_context(),
        active_skills,
        git_available,
    )
    system_message = f"{l1}\n\n{l2}"

    git_snapshot = git_worktree_summary() if git_available else ""
    compaction_note = ""
    if token_count is not None:
        compaction_note = format_compaction_overlay_note(
            CompactionResult(
                messages=[],
                compacted=False,
                token_count=token_count,
                action="none",
                pressure_ratio=token_count / resolve_usable_context_budget(),
                usable_budget=resolve_usable_context_budget(),
            ),
        )

    todos = _sample_todos(mode)
    l3 = "\n\n".join(
        build_working_state_sections(
            mode,
            todos=render_todos(todos),
            session_memory="(loaded from mem_<thread_id>.md at runtime)",
            git_snapshot=git_snapshot,
            compaction_note=compaction_note,
        ).values()
    )
    working_state_tail = _working_state_tail(l3)
    tools_metadata = _render_tools_metadata(tools)
    stable_prefix = f"BOUND TOOLS\n{tools_metadata}\n\nSYSTEM\n{system_message}\n\nMESSAGES\n{query}\n\n"
    full_stream = f"{stable_prefix}{working_state_tail}"
    return {
        "tools": tools,
        "tool_names": tool_names,
        "sticky": sticky,
        "l1": l1,
        "l2": l2,
        "l3": l3,
        "system_message": system_message,
        "query": query,
        "working_state_tail": working_state_tail,
        "tools_metadata": tools_metadata,
        "stable_prefix": stable_prefix,
        "full_stream": full_stream,
        "git_snapshot": git_snapshot,
        "compaction_note": compaction_note,
    }


def _assert_diff_modes(
    *,
    query: str,
    git_available: bool,
    token_count: int | None,
    user_message_count: int,
) -> tuple[int, int, int]:
    plan = _build_mode_parts(
        "plan",
        query=query,
        git_available=git_available,
        token_count=token_count,
        user_message_count=user_message_count,
    )
    act = _build_mode_parts(
        "act",
        query=query,
        git_available=git_available,
        token_count=token_count,
        user_message_count=user_message_count,
    )
    # The full tool set is bound in every mode so the provider prefix cache
    # survives plan<->act switches; schemas and the cached system prefix
    # must therefore be identical across modes.
    if plan["tools_metadata"] != act["tools_metadata"]:
        raise AssertionError("bound tool metadata should be identical across modes")
    if plan["stable_prefix"] != act["stable_prefix"]:
        raise AssertionError("cached system prefix should be identical across modes")
    if "write" not in plan["tool_names"]:
        raise AssertionError("full tool set should be bound in plan mode")
    if plan["tool_names"] != act["tool_names"]:
        raise AssertionError("plan and act mode should expose the same tool set")
    return (len(plan["stable_prefix"]), len(plan["full_stream"]), len(act["full_stream"]))


def _print_mode_preview(
    mode: str,
    *,
    query: str,
    git_available: bool,
    token_count: int | None,
    user_message_count: int,
    compare_tools: bool,
    plan_tool_names: set[str] | None,
) -> set[str]:
    parts = _build_mode_parts(
        mode,
        query=query,
        git_available=git_available,
        token_count=token_count,
        user_message_count=user_message_count,
    )
    tool_names = parts["tool_names"]
    sticky = parts["sticky"]
    l1 = parts["l1"]
    l2 = parts["l2"]
    l3 = parts["l3"]
    system_message = parts["system_message"]
    query = parts["query"]
    working_state_tail = parts["working_state_tail"]
    tools_metadata = parts["tools_metadata"]
    git_snapshot = parts["git_snapshot"]
    compaction_note = parts["compaction_note"]

    mode_style = "magenta" if mode == "plan" else "green"
    console.print(Rule(f"[bold {mode_style}]{mode.upper()} MODE[/]", style=mode_style))

    meta = Table(show_header=False, box=box.SIMPLE, padding=(0, 1))
    meta.add_column("Key", style="dim")
    meta.add_column("Value")
    l1_tokens = _tokens(l1)
    l2_tokens = _tokens(l2)
    prefix_tokens = _tokens(system_message)
    l3_tokens = _tokens(l3)
    tools_metadata_tokens = _tokens(tools_metadata)
    meta.add_row("Active tools", str(len(tool_names)))
    meta.add_row("Tools API metadata tokens", f"~{tools_metadata_tokens:,}")
    meta.add_row("Sticky skills", ", ".join(sorted(sticky)) or "(none)")
    meta.add_row("Git snapshot", git_snapshot or "(not in a repo)")
    meta.add_row("Compaction note", compaction_note or "(none)")
    meta.add_row("Session memory", "(in L3 overlay from mem_<thread_id>.md)")
    over = " [red](over target)[/red]" if prefix_tokens > STATIC_PREFIX_TOKEN_TARGET else ""
    meta.add_row("L0+L1 tokens", f"~{l1_tokens:,}")
    meta.add_row("L2 tokens", f"~{l2_tokens:,}")
    meta.add_row(
        "Static prefix (L0+L1+L2)",
        f"~{prefix_tokens:,} / {STATIC_PREFIX_TOKEN_TARGET:,} target{over}",
    )
    meta.add_row("L3 overlay tokens (ephemeral)", f"~{l3_tokens:,}")
    console.print(meta)
    console.print()

    console.print(
        _panel(
            "Tools — bind_tools() → API tools arg (LangChain convert_to_openai_tool)",
            tools_metadata,
            style="bright_cyan",
        )
    )
    console.print()
    console.print(_panel("L0 + L1 — build_l0() + build_l1() → SystemMessage (cached)", l1, style="blue"))
    console.print()
    console.print(_panel("L2 — build_project_context_block() → SystemMessage (cached)", l2, style="yellow"))
    console.print()
    console.print(_panel("L1 + L2 combined (what _stable_prefix caches)", system_message, style="white"))
    console.print()
    console.print(_panel("L3 — build_working_state_sections() → system-reminder content", l3, style=mode_style))
    console.print()
    console.print(_panel("Original HumanMessage (sent unmodified)", query, style="green"))
    console.print()
    console.print(_panel("L3 system-reminder tail → ephemeral HumanMessage (not persisted)", working_state_tail, style="bold green"))
    console.print()

    if compare_tools and plan_tool_names is not None and mode == "act":
        diff = Table(title="Tool diff (act − plan)", box=box.SIMPLE_HEAD)
        diff.add_column("Only in act", style="green")
        for name in sorted(tool_names - plan_tool_names):
            diff.add_row(name)
        console.print(diff)
        console.print()

    return tool_names


@app.command()
def main(
    mode: str = typer.Option(
        "both",
        "--mode",
        "-m",
        help="Agent mode to preview: plan, act, or both.",
    ),
    query: str = typer.Option(
        "Add retry logic to the web fetch tools",
        "--query",
        "-q",
        help="Example user message used for skill triggers and L3 preview.",
    ),
    git: str = typer.Option(
        "auto",
        "--git",
        help="Git availability: auto (detect), yes, or no.",
    ),
    tokens: int | None = typer.Option(
        None,
        "--tokens",
        "-t",
        help="Simulated context token count for compaction overlay preview.",
    ),
    user_messages: int = typer.Option(
        1,
        "--user-messages",
        "-u",
        help="Simulated human message count (unused; kept for CLI compatibility).",
    ),
    diff_modes: bool = typer.Option(
        False,
        "--diff-modes",
        help="Assert bound tools and system prefix are byte-identical for plan and act mode.",
    ),
) -> None:
    """Print L1/L2/L3 context for the current repository."""
    mode = mode.lower().strip()
    if mode not in {"plan", "act", "both"}:
        raise typer.BadParameter("mode must be plan, act, or both")
    if git not in {"auto", "yes", "no"}:
        raise typer.BadParameter("git must be auto, yes, or no")

    git_available = _resolve_git(git)

    header = Table(title="Context preview", box=box.ROUNDED, show_header=False, padding=(0, 1))
    header.add_column(style="bold cyan")
    header.add_column()
    header.add_row("Query", query)
    header.add_row("Repository", str(ROOT))
    header.add_row("Git", "yes" if git_available else "no")
    header.add_row("Usable compaction budget", f"{resolve_usable_context_budget():,} tokens")
    header.add_row("Reflection token ratio", str(settings.reflection_token_ratio))
    console.print(header)
    console.print()

    if diff_modes:
        try:
            prefix_len, plan_len, act_len = _assert_diff_modes(
                query=query,
                git_available=git_available,
                token_count=tokens,
                user_message_count=user_messages,
            )
        except AssertionError as exc:
            console.print(f"[red]Mode diff failed:[/red] {exc}")
            raise typer.Exit(1) from exc
        console.print(
            f"[green]Mode-stable prefix OK:[/green] {prefix_len:,} bytes "
            f"(plan stream {plan_len:,}, act stream {act_len:,})"
        )
        return

    modes = ["plan", "act"] if mode == "both" else [mode]
    plan_tool_names: set[str] | None = None

    for index, agent_mode in enumerate(modes):
        names = _print_mode_preview(
            agent_mode,
            query=query,
            git_available=git_available,
            token_count=tokens,
            user_message_count=user_messages,
            compare_tools=mode == "both",
            plan_tool_names=plan_tool_names,
        )
        if agent_mode == "plan":
            plan_tool_names = names
        if index < len(modes) - 1:
            console.print()


if __name__ == "__main__":
    app()
