"""End-to-end wiring tests: TuiApp driven by a REAL CodingSession.

The rest of the TUI suite runs against FakeCoding; these tests pin the
actual seam — a TuiApp consuming the SessionEvent stream of a real
``ness_cli.CodingSession`` (real LangGraph run over a bindable fake
chat model), covering the full turn, the cooperative cancel path, and the
resume guard.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from ness_agent import NessAgent, NessAgentOptions, PromptLayers, PromptLayersConfig
from ness_cli import CodingSession

from ness_cli.tui import render
from ness_cli.tui.app import TuiApp


class _BindableFakeModel:
    """bind_tools-capable fake so a REAL langgraph run completes (vs
    FakeListChatModel). Cycles fixed responses."""

    def __init__(self, texts: list[str]) -> None:
        self._texts = list(texts)
        self.calls = 0

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        text = self._texts[min(self.calls, len(self._texts) - 1)]
        self.calls += 1
        return AIMessage(content=text)

    @property
    def model(self):
        return "bindfake"


class _BlockingFakeModel:
    """Fake whose ainvoke blocks until released, for deterministic cancels."""

    def __init__(self, text: str) -> None:
        self._text = text
        self.called = asyncio.Event()
        self.release = asyncio.Event()

    def bind_tools(self, tools):
        return self

    async def ainvoke(self, messages, **kwargs):
        self.called.set()
        await self.release.wait()
        return AIMessage(content=self._text)

    @property
    def model(self):
        return "blockfake"


def _make_agent(tmp_path: Path, model) -> NessAgent:
    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            project_root=tmp_path,
            ness_dir=tmp_path / ".ness",
            auto_save_threads=True,
        ),
    )
    agent.config.thread_store.auto_save = True
    return agent


def _make_tui(coding: CodingSession) -> TuiApp:
    tmp = TemporaryDirectory()
    app = TuiApp(coding, history_path=Path(tmp.name) / "hist")
    app._tmpdir = tmp  # keep the tempdir alive for the test's lifetime
    return app


def _transcript_text(app: TuiApp) -> str:
    return "\n".join(line.text for line in app._lines)


def test_full_turn_end_to_end(tmp_path: Path):
    agent = _make_agent(tmp_path, _BindableFakeModel(["world"]))
    coding = CodingSession(agent, thread_id="t-wire")
    app = _make_tui(coding)
    render.set_sink(app)
    try:
        app.append_user("hello")
        asyncio.run(app._run_turn("hello", []))
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert "hello" in text  # user echo
    assert "world" in text  # assistant panel (non-streaming fake)
    assert app.assistant_history[-1] == "world"
    assert coding.turn_count == 1
    # Durable events: the adapter persisted the user turn and the reply.
    kinds = [e.get("kind") for e in coding.thread_store.load_thread_events("t-wire")]
    assert "user" in kinds
    assert "assistant" in kinds


def test_cooperative_cancel_renders_banner(tmp_path: Path):
    fake = _BlockingFakeModel("late answer")
    agent = _make_agent(tmp_path, fake)
    coding = CodingSession(agent, thread_id="t-cancel")
    app = _make_tui(coding)
    render.set_sink(app)

    async def drive() -> None:
        task = asyncio.create_task(app._run_turn("hello", []))
        # Generous upper bounds: the turn's durable-event append + git
        # checkpoint subprocess run on worker threads before the model call,
        # which can starve the loop under full-suite load. These are caps,
        # not sleeps — fast runs pay nothing.
        await asyncio.wait_for(fake.called.wait(), timeout=15)
        coding.cancel()
        fake.release.set()
        await asyncio.wait_for(task, timeout=15)

    try:
        asyncio.run(drive())
    finally:
        render.set_sink(None)

    assert "Turn interrupted by user." in _transcript_text(app)


def test_resume_unknown_thread_keeps_current(tmp_path: Path):
    agent = _make_agent(tmp_path, _BindableFakeModel(["x"]))
    coding = CodingSession(agent, thread_id="t-live")
    app = _make_tui(coding)
    render.set_sink(app)
    try:
        asyncio.run(app.resume_thread("session-nope"))
    finally:
        render.set_sink(None)

    assert coding.thread_id == "t-live"
    assert "No saved thread" in _transcript_text(app)


def test_resume_replays_saved_thread(tmp_path: Path):
    # Thread "t-old" gets one durable turn; a fresh CodingSession on the same
    # agent backends resumes it through the TUI path.
    agent = _make_agent(tmp_path, _BindableFakeModel(["old reply"]))
    seed = CodingSession(agent, thread_id="t-old")
    asyncio.run(_collect(seed.run_turn("old question")))

    coding = CodingSession(agent, thread_id="t-live")
    app = _make_tui(coding)
    render.set_sink(app)
    try:
        asyncio.run(app.resume_thread("t-old"))
    finally:
        render.set_sink(None)

    assert coding.thread_id == "t-old"
    text = _transcript_text(app)
    assert "old question" in text
    assert "old reply" in text
    assert "Resumed thread" not in text
    assert app.assistant_history[-1] == "old reply"


async def _collect(agen) -> list:
    return [ev async for ev in agen]
