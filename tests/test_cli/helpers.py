from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory

from cli.tui.app import TuiApp


class FakeSession:
    def __init__(self) -> None:
        self.git_available = False
        self.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        self.agent_mode = "act"
        self.should_exit = False
        self.queued_prompt = ""
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

    def toggle_mode(self) -> None:
        self.agent_mode = "plan" if self.agent_mode == "act" else "act"

    def render_header(self) -> None:
        self.header_rendered = True

    def rebuild_graph(self) -> None:
        self.rebuilt = True

    async def refresh_context_snapshot(self) -> None:
        return

    async def run_turn(self, text: str) -> None:
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

    def request_compact(self) -> None:
        self.force_compact = True


Dispatcher = Callable[[FakeSession, str], Awaitable[None]]


def make_app(command_dispatcher: Dispatcher | None = None) -> TuiApp:
    tmp = TemporaryDirectory()
    session = FakeSession()
    kwargs = {"history_path": Path(tmp.name) / "hist"}
    if command_dispatcher is not None:
        kwargs["command_dispatcher"] = command_dispatcher
    app = TuiApp(session, **kwargs)  # type: ignore[arg-type]
    app._tmpdir = tmp
    return app
