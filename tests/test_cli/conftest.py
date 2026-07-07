"""Shared fixtures for the LiteHarness TUI test suite.

Migrated from the legacy ``tests/test_cli/helpers.py`` so tests can take
``make_app`` as a fixture instead of importing a factory function. The
fake implementations mirror the live ``SessionApp`` surface the TUI's
command-dispatch and queue paths depend on, without pulling in the
LangGraph app or model factory.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from cli.app import TuiApp


class _FakeCancelToken:
    """Minimal stand-in for ``cli.session_app.CancelToken`` used by FakeSession.

    Mirrors the trigger/is_set/reset surface the TUI's cancel cascade depends
    on without requiring a real ``SessionApp`` (which would pull in the
    LangGraph app and model factory).
    """

    def __init__(self) -> None:
        self._event = asyncio.Event()

    def trigger(self) -> None:
        self._event.set()

    def is_set(self) -> bool:
        return self._event.is_set()

    def reset(self) -> None:
        self._event.clear()


class FakeSession:
    def __init__(self) -> None:
        self.git_available = False
        self.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        self.agent_mode = "act"
        self.should_exit = False
        self.prompt_queue: list[str] = []
        self.pending_skills: list[str] = []
        self.assistant_history: list[str] = []
        self.last_usage: dict | None = None
        self.turn_count = 0
        self.context_used = 12_400
        self.context_total = 128_000
        self.rebuilt = False
        self.header_rendered = False
        self.force_compact = False
        self.saved = False
        self.resumed_thread_id = ""
        self.rolled_back_seq: int | None = None
        self.cancel_token = _FakeCancelToken()

    def toggle_mode(self) -> None:
        self.agent_mode = "plan" if self.agent_mode == "act" else "act"

    def render_header(self) -> None:
        self.header_rendered = True

    def rebuild_graph(self) -> None:
        self.rebuilt = True

    async def refresh_context_snapshot(self) -> None:
        return

    async def run_turn(self, text: str, image_data_urls: list[str] | None = None) -> None:
        self.turn_count += 1
        self.assistant_history.append(f"echo {text}")

    def save_thread(self) -> str:
        self.saved = True
        return f"Archived thread {self.thread_id}."

    async def reset_thread(self) -> None:
        self.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        self.turn_count = 0
        self.pending_skills.clear()
        self.assistant_history.clear()

    async def resume_thread(self, thread_id: str) -> None:
        self.resumed_thread_id = thread_id
        self.thread_id = thread_id

    async def rollback_to(self, user_seq: int) -> None:
        self.rolled_back_seq = user_seq

    def request_compact(self) -> None:
        self.force_compact = True

    def enqueue_prompt(self, text: str) -> None:
        if text:
            self.prompt_queue.append(text)

    def dequeue_prompt(self) -> str | None:
        if self.prompt_queue:
            return self.prompt_queue.pop(0)
        return None

    def clear_prompt_queue(self) -> int:
        count = len(self.prompt_queue)
        self.prompt_queue.clear()
        return count

    @property
    def queued_prompt(self) -> str:
        return self.prompt_queue[-1] if self.prompt_queue else ""

    @queued_prompt.setter
    def queued_prompt(self, value: str) -> None:
        if value:
            self.prompt_queue = [value]
        else:
            self.prompt_queue.clear()


Dispatcher = Callable[[FakeSession, str], Awaitable[None]]


@pytest.fixture
def make_app() -> Callable[..., TuiApp]:
    """Factory fixture: build a fresh TuiApp backed by a FakeSession.

    Each call returns a TuiApp with its own TemporaryDirectory history path;
    the tempdir's lifetime is tied to the closure (and thus to the test), so
    no explicit cleanup is required.
    """

    def _factory(command_dispatcher: Dispatcher | None = None) -> TuiApp:
        tmp = TemporaryDirectory()
        session = FakeSession()
        kwargs: dict = {"history_path": Path(tmp.name) / "hist"}
        if command_dispatcher is not None:
            kwargs["command_dispatcher"] = command_dispatcher
        app = TuiApp(session, **kwargs)  # type: ignore[arg-type]
        app._tmpdir = tmp
        return app

    return _factory