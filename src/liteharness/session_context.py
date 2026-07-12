from __future__ import annotations

from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from liteharness.options import NessAgentOptions
from liteharness.permissions import PermissionStore
from liteharness.persistence import ThreadStore

if TYPE_CHECKING:
    from liteharness.agent import NessAgentConfig


@dataclass
class SessionContext:
    permissions: PermissionStore
    options: NessAgentOptions
    thread_store: ThreadStore
    ness_dir: Path
    project_root: Path
    agent_config: NessAgentConfig | None = None
    all_skills: dict[str, Any] | None = None


_session_ctx: ContextVar[SessionContext | None] = ContextVar("liteharness_session_context", default=None)


def set_session_context(ctx: SessionContext | None) -> Token:
    return _session_ctx.set(ctx)


def reset_session_context(token: Token) -> None:
    # remove a specific entry from the contextvar stack by its Token
    _session_ctx.reset(token)


def get_session_context() -> SessionContext:
    ctx = _session_ctx.get()
    if ctx is None:
        raise RuntimeError(
            "Session context is not configured. Create a Session (or call set_session_context) "
            "before invoking tools."
        )
    return ctx


def try_get_session_context() -> SessionContext | None:
    return _session_ctx.get()
