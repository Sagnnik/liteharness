"""Construction glue: build a fully-wired coding agent + session.

One place that knows how the coding CLI's moving parts compose: chat models
from :mod:`liteharness_cli.chat_model`, prompt layers / plan-act modes /
task prompts from :mod:`liteharness_cli.prompts`, pricing-aware cost
tracking from :mod:`liteharness_cli.config`, and the SDK's backend
resolution (thread store, permission store, memory store, skill loader,
hook runner, tool registry).

Both the TUI entrypoint (``liteharness_cli.tui.main``) and tests build through here so
there is exactly one wiring recipe::

    from liteharness_cli.factory import build_coding_session

    coding = build_coding_session(
        thread_id="session-abc123",
        approval_handler=render.render_approval_handler,
        question_handler=render.ask_questions,
    )
    async for ev in coding.run_turn("add a rate limiter"):
        ...
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from liteharness import ApprovalHandler, MemoryConfig, NessAgent, NessAgentOptions
from liteharness.workspace import setup_ness_structure

from liteharness_cli.chat_model import (
    active_model_name,
    create_compaction_model,
    create_model,
    create_reflection_model,
)
from liteharness_cli.coding_session import CodingSession
from liteharness_cli.config import (
    context_window_for,
    make_sdk_cost_tracker,
    reload_settings,
    settings,
    settings_field_env_map,
)
from liteharness_cli.config_store import migrate_env_once
from liteharness_cli.paths import (
    NessPaths,
    ensure_global_config,
    ensure_project_runtime,
    resolve_paths,
)
from liteharness_cli.prompts import (
    default_prompt_layers,
    default_aux_prompts,
    plan_act_modes,
)


def prepare_paths(*, project_root: Path | None = None) -> NessPaths:
    """Resolve paths and ensure global config + project runtime dirs exist.

    On first run, imports any known keys from a project ``.env`` into the
    global ``configs.json`` / ``secrets.json`` (once; the ``.env`` file is
    left untouched), then reloads settings so the migrated values apply.
    """
    paths = resolve_paths(
        project_root=project_root or Path.cwd(),
        ness_dir=settings.ness_dir,
    )
    ensure_global_config(paths)
    migrated = migrate_env_once(
        Path.cwd() / ".env",
        config_dir=paths.config_dir,
        field_for_alias=settings_field_env_map(),
    )
    if migrated:
        reload_settings()
    ensure_project_runtime(paths)
    return paths


def build_coding_agent(
    *,
    thread_id: str,
    yolo: bool = False,
    approval_handler: ApprovalHandler | None = None,
    question_handler: Any = None,
    l2_context: str | None = None,
    paths: NessPaths | None = None,
    **agent_kwargs: Any,
) -> NessAgent:
    """Build a :class:`NessAgent` with the coding defaults.

    ``approval_handler`` / ``question_handler`` are threaded straight into
    the agent spec — pass the interactive handlers *here* (not by mutating
    the config afterwards) so the SDK's event bridges wrap them at first
    session creation. Extra keyword arguments override the coding defaults
    (e.g. ``tools=``, ``overlay=``, ``tracing=``).
    """
    if paths is None:
        paths = prepare_paths()
    kwargs: dict[str, Any] = {
        "model": create_model(thread_id),
        "compaction_model": create_compaction_model(thread_id),
        "reflection_model": create_reflection_model(thread_id),
        "prompt": default_prompt_layers(l2_context=l2_context),
        "aux_prompts": default_aux_prompts(),
        "modes": plan_act_modes(plans_dir=paths.plans_dir),
        "memory": MemoryConfig(
            user_memory=paths.user_file,
            session_memory_dir=paths.sessions_dir,
        ),
        "hooks_config": paths.ness_dir / "hooks.json",
        "skills_dir": paths.ness_dir / "skills",
        "options": NessAgentOptions(
            context_window=context_window_for(active_model_name()),
            compaction_token_budget=settings.compaction_token_budget,
            compaction_output_reserve=settings.compaction_output_reserve,
            compaction_input_reserve=settings.compaction_input_reserve,
            enable_approval=settings.enable_approval and not yolo,
            yolo_mode=yolo,
            auto_save_threads=settings.auto_save_threads,
            reflection_token_ratio=settings.reflection_token_ratio,
            session_end_reflection=settings.session_end_reflection,
            format_on_write=settings.format_on_write,
            exa_api_key=settings.exa_api_key,
            project_root=paths.project_root,
            ness_dir=paths.ness_dir,
        ),
        "approval_handler": approval_handler,
        "question_handler": question_handler,
        "cost_tracker": make_sdk_cost_tracker(),
    }
    kwargs.update(agent_kwargs)
    return NessAgent(**kwargs)


def build_coding_session(
    *,
    thread_id: str,
    mode: str = "act",
    vision: bool | None = None,
    git_available: bool | None = None,
    metadata: dict[str, Any] | None = None,
    paths: NessPaths | None = None,
    **agent_kwargs: Any,
) -> CodingSession:
    """Build a :class:`CodingSession` on a fresh coding agent.

    ``agent_kwargs`` are forwarded to :func:`build_coding_agent` (handlers,
    prompt overrides, tracing, ...).
    """
    agent = build_coding_agent(thread_id=thread_id, paths=paths, **agent_kwargs)
    return CodingSession(
        agent,
        thread_id=thread_id,
        mode=mode,
        vision=vision,
        git_available=git_available,
        metadata=metadata,
    )
