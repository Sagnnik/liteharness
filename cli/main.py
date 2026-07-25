"""LiteHarness CLI entrypoint.

A clean, scrollable Rich + prompt_toolkit CLI. Run with:

    uv run python -m cli.main
    # or
    uv run python cli/main.py

The TUI is wired directly to the SDK stack: a
:class:`~liteharness_cli.CodingSession` built via
:func:`liteharness_cli.factory.build_coding_session`, an SDK
:class:`~liteharness.mcp.MCPManager`, and the SDK tool registry. The old
root-level monolith modules are no longer imported here.
"""

from __future__ import annotations

import asyncio
import os
import sys
import uuid
from pathlib import Path

# Allow running both as `python -m cli.main` and `python cli/main.py` by making
# sure the project root (which holds the cli package) is importable.
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
    from liteharness_cli.worktree import WorktreeError, ensure_worktree

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

from liteharness.compaction import resolve_token_count
from liteharness.mcp import MCPManager
from liteharness.tools import is_git_repo, register_dynamic_tools, set_mcp_catalog
from liteharness_cli.chat_model import (
    ModelOverrides,
    configure_model,
    validate_reasoning_effort_for_model,
)
from liteharness_cli.config import REASONING_EFFORTS, settings
from liteharness_cli.factory import build_coding_session

from cli import render
from cli.app import TuiApp
from cli.theme import build_console

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
    model: str = typer.Option(
        None, "--model", help="Chat model name (overrides MODEL_NAME)"
    ),
    reflection_model: str = typer.Option(
        None, "--reflection-model", help="Reflection model name"
    ),
    api_key: str = typer.Option(None, "--api-key", help="OpenAI-compatible API key"),
    base_url: str = typer.Option(None, "--base-url", help="OpenAI-compatible base URL"),
    session_id: str = typer.Option(
        None, "--openrouter-session-id", help="Stable prompt-cache session id"
    ),
    reasoning_effort: str = typer.Option(
        None,
        "--reasoning-effort",
        help="OpenRouter reasoning effort: none, minimal, low, medium, high, xhigh, max",
    ),
    worktree: str = typer.Option(
        None, "--worktree", "-w", help="Run inside an isolated git worktree"
    ),
    resume: str = typer.Option(
        None,
        "--resume",
        help="Resume a saved thread id at startup (loads prior conversation into the transcript)",
    ),
) -> None:
    """Start an interactive LiteHarness session."""
    configure_model(
        _overrides(
            model, reflection_model, api_key, base_url, session_id, reasoning_effort
        )
    )
    asyncio.run(_main(resume_thread_id=resume or None))


def _render_mcp_startup(mcp: MCPManager) -> None:
    message, level = mcp.startup_summary()
    if level != "warn":
        return
    render.render_warning(message + "  (/mcp for details)")


_STATIC_PREFIX_TOKEN_TARGET = 7000


def _check_prompt_budget(coding, git_available: bool) -> str | None:
    """Soft-warn when the cached static prefix (L0+L1+L2) exceeds the token target."""
    from langchain_core.messages import SystemMessage

    cfg = coding.agent.config
    prefix = cfg.prompts.build_stable_prefix(
        cfg.tool_registry.active_tools,
        user_memory=cfg.memory_store.load_user(),
        project_memory=cfg.memory_store.load_project(),
        skill_catalog=cfg.skill_loader.render_catalog(cfg.skill_loader.load()),
        git_available=git_available,
        metadata={},
        tool_catalog_groups=cfg.tool_registry.tool_catalog_groups(),
        mcp_catalog=cfg.tool_registry.mcp_catalog(),
        deferred_mcp=cfg.tool_registry.deferred_mcp_summary(),
    ).strip()
    tokens = resolve_token_count(
        [SystemMessage(content=prefix)], known_input_tokens=None
    )
    if tokens > _STATIC_PREFIX_TOKEN_TARGET:
        return (
            f"Static prompt prefix (L0+L1+L2) is ~{tokens:,} tokens "
            f"(> {_STATIC_PREFIX_TOKEN_TARGET:,} target). Consider trimming NESS.md, "
            "its @includes, or USER.md."
        )
    return None


async def _main(*, resume_thread_id: str | None = None) -> None:
    git_available = is_git_repo()

    mcp = MCPManager(project_root=Path.cwd())
    await mcp.start()

    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    coding = build_coding_session(
        thread_id=thread_id,
        vision=settings.supports_vision,
        git_available=git_available,
        approval_handler=render.ask_approval,
        question_handler=render.ask_questions,
    )
    coding.perms.clear_session_rules()

    # Register MCP tools as known but deferred: nothing is bound until
    # /mcp (session registry) or the model-facing add_tools (module-level
    # bridge) activates them.
    mcp_tools = list(mcp.tools.values())
    coding.tool_registry.register_dynamic(mcp_tools)
    coding.tool_registry.set_mcp_catalog(mcp.catalog())
    register_dynamic_tools(mcp_tools)
    set_mcp_catalog(mcp.catalog())

    ui = TuiApp(
        coding,
        mcp=mcp,
        history_path=Path(settings.ness_dir) / "cli_history",
    )
    render.set_sink(ui)
    # The startup header is rendered by ``TuiApp`` once the transcript pane's
    # real width is known (see ``TuiApp._start_initial_header_task``), so the
    # pre-wrapped TranscriptLines aren't built against a stale fallback width
    # and re-wrap into a half-screen artifact on first render.

    worktree_name = os.environ.get("LITEHARNESS_WORKTREE")
    if worktree_name:
        worktree_path = os.environ.get("LITEHARNESS_WORKTREE_PATH", os.getcwd())
        render.render_notice(
            f"worktree: {worktree_name} @ {worktree_path}",
            title="worktree",
        )

    warning = coding.memory_store.check_health()
    if warning:
        render.render_warning(warning)
    budget_warning = _check_prompt_budget(coding, git_available)
    if budget_warning:
        render.render_warning(budget_warning)
    _render_mcp_startup(mcp)

    try:
        await ui.run_async(resume_thread_id=resume_thread_id)
    finally:
        resume_thread_id = None
        try:
            await coding.finalize_reflection()
            resume_thread_id = (
                coding.thread_id
                if (settings.auto_save_threads and coding.turn_count > 0)
                else None
            )
            coding.save_thread()
        except Exception as exc:
            render.render_warning(f"Archive skipped: {exc}")
        await mcp.stop()
        render.set_sink(None)
        # The fullscreen Textual TUI has exited, so the transcript sink is no
        # longer on screen. Print the session summary straight to stdout via
        # a standalone Rich console rather than routing through the (dead)
        # in-app transcript widget, otherwise it is silently dropped.
        from rich.panel import Panel

        report = coding.cost_tracker.report()
        if resume_thread_id:
            report += f"\nResume:  liteharness --resume {resume_thread_id}"
        build_console(file=sys.stdout).print(
            Panel(
                report,
                title="session summary",
                style="usage.value",
                border_style="usage.value",
            )
        )


def main() -> None:
    app()


if __name__ == "__main__":
    main()
