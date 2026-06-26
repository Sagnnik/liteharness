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

from config import cost_tracker, settings
from memory import check_ness_health
from mcp_client import mcp_manager
from model import ModelOverrides, active_model_name, configure_model
from permissions import clear_session_rules
from tools import is_git_repo, register_dynamic_tools, set_mcp_catalog

from cli import render
from cli.commands import dispatch
from cli.prompt import PromptController
from cli.session_app import SessionApp

app = typer.Typer(add_completion=False, help="LiteHarness agent CLI")


def _overrides(
    model: str | None,
    reflection_model: str | None,
    api_key: str | None,
    base_url: str | None,
    session_id: str | None,
) -> ModelOverrides | None:
    fields = {
        "model_name": model,
        "reflection_model_name": reflection_model,
        "openai_api_key": api_key,
        "openai_base_url": base_url,
        "openrouter_session_id": session_id,
    }
    active = {key: value for key, value in fields.items() if value is not None}
    return ModelOverrides(**active) if active else None


@app.command()
def run(
    model: str = typer.Option(None, "--model", help="Chat model name (overrides MODEL_NAME)"),
    reflection_model: str = typer.Option(None, "--reflection-model", help="Reflection model name"),
    api_key: str = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    session_id: str = typer.Option(None, "--openrouter-session-id", help="Stable prompt-cache session id"),
    worktree: str = typer.Option(None, "--worktree", "-w", help="Run inside an isolated git worktree"),
) -> None:
    """Start an interactive LiteHarness session."""
    configure_model(_overrides(model, reflection_model, api_key, base_url, session_id))
    asyncio.run(_main())


def _render_mcp_startup() -> None:
    message, level = mcp_manager.startup_summary()
    style = {"ok": "accent", "warn": "warning", "none": "muted"}.get(level, "muted")
    hint = "" if level == "ok" else "  (/mcp for details)"
    render.console.print(render.Text(message + hint, style=style))


async def _main() -> None:
    git_available = is_git_repo()
    clear_session_rules()

    await mcp_manager.start()
    # Register MCP tools as known (so they can be activated later) but leave them
    # deferred: nothing is bound until search_tools/add_tools or /mcp loads them.
    register_dynamic_tools(mcp_manager.tools.values())
    set_mcp_catalog(mcp_manager.catalog())

    session = SessionApp(git_available=git_available)
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
    _render_mcp_startup()

    controller = PromptController(
        get_mode=lambda: session.agent_mode,
        get_model=active_model_name,
        get_usage=lambda: session.last_usage,
        toggle_mode=session.toggle_mode,
        history_path=Path(settings.ness_dir) / "cli_history",
    )

    try:
        while not session.should_exit:
            queued = session.queued_prompt
            session.queued_prompt = ""
            if queued:
                user_input = queued
            else:
                try:
                    user_input = (await controller.ask()).strip()
                except (EOFError, KeyboardInterrupt):
                    break

            if not user_input:
                continue

            if user_input.startswith("/"):
                await dispatch(session, user_input)
                if session.should_exit:
                    break
                continue

            try:
                await session.run_turn(user_input)
            except KeyboardInterrupt:
                render.render_warning("Turn interrupted.")
    finally:
        try:
            await session.finalize_reflection()
            session.save_thread()
        except Exception as exc:
            render.render_warning(f"Archive skipped: {exc}")
        await mcp_manager.stop()
        render.render_panel_text(cost_tracker.report(), title="session summary", style="usage.value")


if __name__ == "__main__":
    app()
