"""Shared helpers for tests that exercise SDK tools via SessionContext."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import Token
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

from langchain_core.language_models.fake_chat_models import FakeListChatModel

from liteharness.agent import AgentSpec, NessAgentConfig
from liteharness.context.layers import PromptLayersConfig
from liteharness.options import NessAgentOptions
from liteharness.permissions import PermissionStore
from liteharness.persistence import ThreadStore
from liteharness.session_context import SessionContext, reset_session_context, set_session_context


def make_session_context(
    project_root: Path,
    *,
    format_on_write: bool = False,
    exa_api_key: str | None = None,
    auto_save: bool = False,
    enable_approval: bool = False,
    agent_config: NessAgentConfig | None = None,
    tools: list[str] | None = None,
) -> SessionContext:
    """Build a SessionContext rooted at *project_root* (creates ``.ness``)."""
    ness = project_root / ".ness"
    ness.mkdir(parents=True, exist_ok=True)
    options = NessAgentOptions(
        project_root=project_root,
        ness_dir=ness,
        format_on_write=format_on_write,
        exa_api_key=exa_api_key,
        enable_approval=enable_approval,
        auto_save_threads=auto_save,
    )
    perms = PermissionStore(ness_dir=ness, project_root=project_root)
    store = ThreadStore(threads_dir=ness / "threads", auto_save=auto_save)
    cfg = agent_config
    if cfg is None and tools is not None:
        cfg = resolve_minimal_agent_config(project_root, ness, tools=tools, options=options)
    return SessionContext(
        permissions=perms,
        options=options,
        thread_store=store,
        ness_dir=ness,
        project_root=project_root,
        agent_config=cfg,
    )


def resolve_minimal_agent_config(
    project_root: Path,
    ness_dir: Path,
    *,
    tools: list[str] | None = None,
    options: NessAgentOptions | None = None,
    model: Any | None = None,
) -> NessAgentConfig:
    """Resolve a minimal NessAgentConfig for subagent / graph-adjacent tests."""
    opts = options or NessAgentOptions(
        project_root=project_root,
        ness_dir=ness_dir,
        format_on_write=False,
        enable_approval=False,
        auto_save_threads=False,
    )
    spec = AgentSpec(
        model=model or FakeListChatModel(responses=["ok"]),
        prompt=PromptLayersConfig(l0="test harness"),
        tools=tools if tools is not None else ["read", "grep", "glob"],
        options=opts,
    )
    return NessAgentConfig.resolve(spec)


@contextmanager
def installed_session_context(
    project_root: Path,
    **kwargs: Any,
) -> Iterator[SessionContext]:
    """Install a SessionContext for the duration of the block."""
    ctx = make_session_context(project_root, **kwargs)
    token = set_session_context(ctx)
    try:
        yield ctx
    finally:
        reset_session_context(token)


def set_exa_key(ctx: SessionContext, key: str | None) -> None:
    """Mutate the installed context's Exa API key (and clear provider cache externally)."""
    ctx.options = replace(ctx.options, exa_api_key=key)


class SessionContextTestMixin:
    """unittest mixin: install/uninstall SessionContext around each test."""

    root: Path
    ctx: SessionContext
    _ctx_token: Token

    def install_ctx(self, root: Path, **kwargs: Any) -> SessionContext:
        self.root = root
        self.ctx = make_session_context(root, **kwargs)
        self._ctx_token = set_session_context(self.ctx)
        return self.ctx

    def uninstall_ctx(self) -> None:
        reset_session_context(self._ctx_token)
