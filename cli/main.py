"""LiteHarness CLI entrypoint.

A clean, scrollable Rich + prompt_toolkit CLI. Run with:

    uv run python -m cli.main
    # or
    uv run python cli/main.py
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

# Allow running both as `python -m cli.main` and `python cli/main.py` by making
# sure the project root (which holds agent.py, config.py, ...) is importable.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _bootstrap_worktree() -> None:
    argv = sys.argv[1:]
    name = None
    for i, arg in enumerate(argv):
        if arg in ("--worktree", "-w") and i + 1 < len(argv):
            name = argv[i + 1]
        elif arg.startswith("--worktree="):
            name = arg.split("=", 1)[1]
    if not name:
        return
    from worktree import WorktreeError, ensure_worktree

    try:
        path = ensure_worktree(name)
    except WorktreeError as exc:
        print(f"worktree error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    os.chdir(path)
    os.environ["LITEHARNESS_WORKTREE"] = name
    os.environ["LITEHARNESS_WORKTREE_PATH"] = str(path)


_bootstrap_worktree()

import typer

from config import REASONING_EFFORTS, cost_tracker, settings
from memory import check_ness_health
from mcp_client import mcp_manager
from model import ModelOverrides, configure_model, validate_reasoning_effort_for_model
from permissions import clear_session_rules
from tools import is_git_repo, register_dynamic_tools, set_mcp_catalog

from cli import render
from cli.session_app import SessionApp
from cli.tui import TuiApp

app = typer.Typer(add_completion=False, help="LiteHarness agent CLI")


def _overrides(
    model: str | None,
    reflection_model: str | None,
    api_key: str | None,
    base_url: str | None,
    session_id: str | None,
    reasoning_effort: str | None,
) -> ModelOverrides | None:
    fields = {
        "model_name": model,
        "reflection_model_name": reflection_model,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "openrouter_session_id": session_id,
        "reasoning_effort": reasoning_effort,
    }
    if reasoning_effort is not None:
        if reasoning_effort not in REASONING_EFFORTS:
            allowed = ", ".join(REASONING_EFFORTS)
            raise typer.BadParameter(f"reasoning effort must be one of: {allowed}")
        target_model = model or settings.model_name
        try:
            validate_reasoning_effort_for_model(target_model, reasoning_effort)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    active = {key: value for key, value in fields.items() if value is not None}
    return ModelOverrides(**active) if active else None


@app.command()
def run(
    model: str = typer.Option(None, "--model", help="Chat model name (overrides MODEL_NAME)"),
    reflection_model: str = typer.Option(None, "--reflection-model", help="Reflection model name"),
    api_key: str = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    session_id: str = typer.Option(None, "--openrouter-session-id", help="Stable prompt-cache session id"),
    reasoning_effort: str = typer.Option(
        None,
        "--reasoning-effort",
        help="OpenRouter reasoning effort: none, minimal, low, medium, high, xhigh, max",
    ),
    worktree: str = typer.Option(None, "--worktree", "-w", help="Run inside an isolated git worktree"),
) -> None:
    """Start an interactive LiteHarness session."""
    configure_model(_overrides(model, reflection_model, api_key, base_url, session_id, reasoning_effort))
    asyncio.run(_main())


def _render_mcp_startup() -> None:
    message, level = mcp_manager.startup_summary()
    hint = "" if level == "ok" else "  (/mcp for details)"
    if level == "warn":
        render.render_warning(message + hint)
    else:
        render.render_notice(message + hint, title="mcp" if level == "ok" else None)


_STATIC_PREFIX_TOKEN_TARGET = 7000


def _check_prompt_budget(git_available: bool) -> str | None:
    """Soft-warn when the cached static prefix (L0+L1+L2) exceeds the token target."""
    from langchain_core.messages import SystemMessage

    from compaction import resolve_token_count
    from context import (
        DEFAULT_PERSONA,
        build_l0,
        build_l1,
        build_project_context_block,
    )
    from memory import load_ness_memory, load_repo_context, load_user_memory
    from skill_loader import load_skills, render_skill_catalog
    from tools import select_tools_for_session

    tools = select_tools_for_session()
    catalog = render_skill_catalog(load_skills())
    prefix = "\n\n".join(
        [
            build_l0(tools),
            build_l1(DEFAULT_PERSONA, tools, load_user_memory(), load_ness_memory(), catalog),
            build_project_context_block(load_repo_context(), [], git_available),
        ]
    ).strip()
    tokens = resolve_token_count([SystemMessage(content=prefix)], known_input_tokens=None)
    if tokens > _STATIC_PREFIX_TOKEN_TARGET:
        return (
            f"Static prompt prefix (L0+L1+L2) is ~{tokens:,} tokens "
            f"(> {_STATIC_PREFIX_TOKEN_TARGET:,} target). Consider trimming NESS.md, "
            "its @includes, or USER.md."
        )
    return None


async def _main() -> None:
    git_available = is_git_repo()
    clear_session_rules()

    await mcp_manager.start()
    # Register MCP tools as known (so they can be activated later) but leave them
    # deferred: nothing is bound until search_tools/add_tools or /mcp loads them.
    register_dynamic_tools(mcp_manager.tools.values())
    set_mcp_catalog(mcp_manager.catalog())

    session = SessionApp(git_available=git_available)
    ui = TuiApp(
        session,
        history_path=Path(settings.ness_dir) / "cli_history",
    )
    render.set_sink(ui)
    session.render_header()

    worktree_name = os.environ.get("LITEHARNESS_WORKTREE")
    if worktree_name:
        worktree_path = os.environ.get("LITEHARNESS_WORKTREE_PATH", os.getcwd())
        render.render_notice(
            f"worktree: {worktree_name} @ {worktree_path}",
            title="worktree",
        )

    warning = check_ness_health()
    if warning:
        render.render_warning(warning)
    budget_warning = _check_prompt_budget(git_available)
    if budget_warning:
        render.render_warning(budget_warning)
    _render_mcp_startup()

    try:
        await ui.run_async()
    finally:
        try:
            await session.finalize_reflection()
            session.save_thread()
        except Exception as exc:
            render.render_warning(f"Archive skipped: {exc}")
        await mcp_manager.stop()
        render.render_panel_text(cost_tracker.report(), title="session summary", style="usage.value")
        render.set_sink(None)


if __name__ == "__main__":
    app()
