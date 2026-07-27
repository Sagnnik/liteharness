"""Shared fixtures for the LiteHarness TUI test suite.

The TUI is wired directly to a ``liteharness_cli.CodingSession``: TuiApp
owns the TUI-side session state (prompt queue, exit flag, staged skills,
assistant history) and consumes the coding session's SessionEvent stream.
The fakes below mirror the ``CodingSession`` surface the TUI and slash
commands touch, without pulling in the LangGraph app or model factory.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest

from liteharness.types import SessionEvent

from cli.app import TuiApp


class _FakeThreadStore:
    def __init__(self) -> None:
        self.events: dict[str, list[dict]] = {}
        self.archived: list[str] = []

    def load_thread_events(self, thread_id: str) -> list[dict]:
        return list(self.events.get(thread_id, []))

    def list_threads(self, n: int = 10) -> list[dict]:
        return []

    def first_user_message(self, thread_id: str) -> str | None:
        return None

    def list_user_turns(self, thread_id: str) -> list[dict]:
        return []

    def archive_thread(self, thread_id: str) -> str:
        self.archived.append(thread_id)
        return f"Archived thread {thread_id}."


class _FakeMemoryStore:
    def __init__(self) -> None:
        self.ness_file = Path(".ness") / "NESS.md"
        self.user_file = Path(".ness") / "USER.md"
        self._session_raw: dict[str, str] = {}

    @property
    def disabled(self) -> bool:
        return False

    def load_project(self) -> str:
        return ""

    def load_user(self) -> str:
        return ""

    def load_session(self, thread_id: str) -> str:
        return ""

    def append_project(self, text: str) -> str:
        return "Updated .ness/NESS.md"

    def append_user(self, text: str) -> str:
        return "Updated USER.md"

    def append_session_bullets(self, thread_id: str, bullets: list[str]) -> bool:
        return False

    def write_project(self, text: str, overwrite: bool = False) -> str:
        return "Wrote .ness/NESS.md"

    def write_user(self, text: str, overwrite: bool = False) -> str:
        return "Wrote USER.md"

    def read_session_raw(self, thread_id: str) -> str:
        return self._session_raw.get(thread_id, "")

    def write_session_raw(self, thread_id: str, text: str) -> None:
        if text:
            self._session_raw[thread_id] = text
        else:
            self._session_raw.pop(thread_id, None)

    def check_health(self) -> str | None:
        return None


class _FakePerms:
    def list_rules(self) -> str:
        return "(no rules)"

    def persist_rule(self, rule: str, bucket: str, scope: str = "always") -> None:
        return None

    def remove_rule(self, bucket: str, index: int) -> str:
        return f"rule #{index}"

    def clear_session_rules(self) -> None:
        return None


class _FakeHookRunner:
    def describe(self) -> str:
        return "(no hooks)"


class _FakeSkillLoader:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def load(self) -> dict:
        return {}


class _FakeToolRegistry:
    def activate_mcp(self, names) -> tuple[list[str], list[str]]:
        return (list(names), [])

    def tool_names(self) -> list[str]:
        return []


class _FakeCostTracker:
    """Flat-attribute stand-in for the SDK CostTracker scalar surface."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.uncached_input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.calls = 0

    def report(self) -> str:
        return "Calls: 0"


class FakeCoding:
    """Stand-in for ``liteharness_cli.CodingSession``.

    Mirrors the surface TuiApp and the slash commands use: backend stores,
    mode/context properties, the run_turn event stream (scripted via
    :meth:`queue_events`), and thread management. ``run_turn`` yields an
    ``assistant_final`` echo by default so tests can drive a turn without
    scripting events.
    """

    def __init__(self) -> None:
        self.thread_id = f"session-{uuid.uuid4().hex[:8]}"
        self.mode = "act"
        self.turn_count = 0
        self.context_used = 12_400
        self.context_total = 128_000
        self.ness_dir = Path(".ness")
        self.project_root = Path.cwd()
        self.thread_store = _FakeThreadStore()
        self.cost_tracker = _FakeCostTracker()
        self.permission_store = _FakePerms()
        self.memory_store = _FakeMemoryStore()
        self.hook_runner = _FakeHookRunner()
        self.skill_loader = _FakeSkillLoader()
        self.tool_registry = _FakeToolRegistry()
        self.agent = SimpleNamespace(
            config=SimpleNamespace(model=SimpleNamespace())
        )
        self._pending_skills: list[str] = []

        self.cancelled = False
        self.resumed: list[str] = []
        self.reset_ids: list[str] = []
        self.rolled_back_seq: int | None = None
        self.compact_requested = False
        self.saved = False
        self.reloaded = False
        self._events: list[SessionEvent] = []

    # --- scripting ---------------------------------------------------------
    def queue_events(self, *events: SessionEvent) -> None:
        """Script the events the next ``run_turn`` will yield (replacing the
        default echo for that turn)."""
        self._events.extend(events)

    # --- the turn -----------------------------------------------------------
    async def run_turn(
        self,
        text: str,
        *,
        images: list[str] | None = None,
        active_skills: list[str] | None = None,
        mode: str | None = None,
    ):
        self.turn_count += 1
        events, self._events = self._events, []
        if not events:
            events = [SessionEvent("assistant_final", {"content": f"echo {text}"})]
        for ev in events:
            yield ev

    # --- control -------------------------------------------------------------
    def cancel(self) -> None:
        self.cancelled = True

    def is_cancelled(self) -> bool:
        return self.cancelled

    def toggle_mode(self) -> str:
        self.mode = "plan" if self.mode == "act" else "act"
        return self.mode

    def request_compact(self) -> None:
        self.compact_requested = True

    def active_skills(self, names: list[str]) -> None:
        self._pending_skills = list(names)

    def stage_skills(self, names) -> None:
        pending = list(self._pending_skills)
        seen = set(pending)
        for name in names:
            n = str(name).strip()
            if n and n not in seen:
                pending.append(n)
                seen.add(n)
        self._pending_skills = pending

    def save_thread(self) -> str:
        self.saved = True
        return self.thread_store.archive_thread(self.thread_id)

    def reload_model(self) -> None:
        self.reloaded = True

    async def finalize_reflection(self) -> None:
        return None

    async def refresh_context_snapshot(self) -> dict:
        return {}

    async def get_todos(self) -> list[dict]:
        return []

    # --- thread management ----------------------------------------------------
    async def resume(self, thread_id: str, *, replay_cost: bool = True) -> bool:
        self.resumed.append(thread_id)
        self.thread_id = thread_id
        return True

    async def reset(self, thread_id: str) -> None:
        self.reset_ids.append(thread_id)
        self.thread_id = thread_id
        self.turn_count = 0

    async def rollback_to(self, user_seq: int) -> str:
        self.rolled_back_seq = user_seq
        return f"Rolled back to turn @ seq {user_seq}."


Dispatcher = Callable[[TuiApp, str], Awaitable[None]]


@pytest.fixture
def make_app() -> Callable[..., TuiApp]:
    """Factory fixture: build a fresh TuiApp backed by a FakeCoding.

    Each call returns a TuiApp with its own TemporaryDirectory history path;
    the tempdir's lifetime is tied to the closure (and thus to the test), so
    no explicit cleanup is required.
    """

    def _factory(command_dispatcher: Dispatcher | None = None) -> TuiApp:
        tmp = TemporaryDirectory()
        coding = FakeCoding()
        kwargs: dict = {"history_path": Path(tmp.name) / "hist"}
        if command_dispatcher is not None:
            kwargs["command_dispatcher"] = command_dispatcher
        app = TuiApp(coding, **kwargs)  # type: ignore[arg-type]
        app._tmpdir = tmp
        return app

    return _factory
