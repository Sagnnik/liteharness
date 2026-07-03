from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace
from typing import Any

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from cli import render
from cli.session_app import CancelToken, SessionApp


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

class _FakeSnapshot:
    def __init__(self, messages: list[Any]) -> None:
        self.values = {"messages": messages}


class _FakeApp:
    """Stand-in for the compiled LangGraph app used by SessionApp.run_turn.

    Lets the test inject a scripted event sequence for ``astream_events`` and
    a canned snapshot for ``aget_state`` so the cancel-cleanup logic can be
    exercised end-to-end without an LLM round-trip. ``trigger_token_after``
    arms the SessionApp's cancel token after the Nth event is yielded so the
    loop's next ``is_set()`` check breaks out — this mirrors a real user
    pressing Ctrl+C mid-stream.
    """

    def __init__(
        self,
        events: list[dict],
        *,
        state_messages: list[Any] | None = None,
        trigger_token_after: int | None = None,
        token: CancelToken | None = None,
    ) -> None:
        self._events = list(events)
        self._state_messages = state_messages or []
        self._trigger_token_after = trigger_token_after
        self._token = token
        self.updates: list[dict] = []

    async def astream_events(self, payload, *, config=None, version="v2"):
        if self._trigger_token_after == 0 and self._token is not None:
            self._token.trigger()
        for index, event in enumerate(self._events):
            yield event
            if self._trigger_token_after is not None and self._token is not None:
                if index + 1 >= self._trigger_token_after:
                    self._token.trigger()

    async def aget_state(self, config):
        return _FakeSnapshot(self._state_messages)

    async def aupdate_state(self, config, updates):
        self.updates.append(updates)


def _set_event_loop_for_tests() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


def _make_session_app() -> SessionApp:
    os.environ.setdefault("OPENROUTER_API_KEY", "test")
    return SessionApp(git_available=False)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


# --------------------------------------------------------------------------- #
# CancelToken behaviour
# --------------------------------------------------------------------------- #

def test_cancel_token_is_set_reset_and_wait():
    token = CancelToken()
    assert token.is_set() is False
    token.trigger()
    assert token.is_set() is True
    token.reset()
    assert token.is_set() is False
    token.trigger()
    asyncio.run(token.wait())


# --------------------------------------------------------------------------- #
# 1. Cancel mid-stream records the partial assistant text as interrupted
# --------------------------------------------------------------------------- #

def test_cancel_mid_stream_records_interrupted_text(monkeypatch):
    app = _make_session_app()

    events = [
        {"event": "on_chat_model_start", "name": "agent"},
        {"event": "on_chat_model_stream", "name": "agent",
         "data": {"chunk": SimpleNamespace(content="Partial assistant text")}},
    ]
    fake = _FakeApp(events, trigger_token_after=2, token=app.cancel_token)
    monkeypatch.setattr(app, "app", fake)

    render.set_sink(None)
    try:
        _run(app.run_turn("do something"))
    finally:
        render.set_sink(None)

    assert app.assistant_history, "expected the partial assistant text to be recorded"
    assert app.assistant_history[-1].endswith("[interrupted]")
    assert "Partial assistant text" in app.assistant_history[-1]


# --------------------------------------------------------------------------- #
# 2. Cancel mid-tool-call injects synthetic failed ToolMessages
# --------------------------------------------------------------------------- #

def test_cancel_mid_tool_calls_injects_failed_tool_messages(monkeypatch):
    app = _make_session_app()

    pending_ai = AIMessage(
        content="",
        tool_calls=[
            {"name": "shell", "args": {"cmd": "ls"}, "id": "call-1", "type": "tool_call"},
            {"name": "read_file", "args": {"path": "x"}, "id": "call-2", "type": "tool_call"},
        ],
    )
    state_messages = [HumanMessage(content="run"), pending_ai]
    fake = _FakeApp(
        events=[],
        state_messages=state_messages,
        trigger_token_after=0,
        token=app.cancel_token,
    )
    monkeypatch.setattr(app, "app", fake)

    render.set_sink(None)
    try:
        _run(app.run_turn("run"))
    finally:
        render.set_sink(None)

    assert len(fake.updates) == 1
    injected = fake.updates[0]["messages"]
    assert len(injected) == 2
    ids = {m.tool_call_id for m in injected}
    assert ids == {"call-1", "call-2"}
    for msg in injected:
        assert isinstance(msg, ToolMessage)
        assert msg.content == "Tool execution interrupted"


# --------------------------------------------------------------------------- #
# 3. Cancel skips success-path renders but autosaves plan
# --------------------------------------------------------------------------- #

def test_cancel_skips_usage_footer_and_todos_but_autosaves_plan(monkeypatch):
    app = _make_session_app()
    fake = _FakeApp(events=[], trigger_token_after=0, token=app.cancel_token)
    monkeypatch.setattr(app, "app", fake)

    called = {"footer": 0, "todos": 0, "autosave": 0}

    monkeypatch.setattr(render, "render_usage_footer", lambda *_a, **_k: called.__setitem__("footer", called["footer"] + 1))
    monkeypatch.setattr(render, "render_todos", lambda *_a, **_k: called.__setitem__("todos", called["todos"] + 1))
    monkeypatch.setattr(app, "_autosave_plan_turn", lambda: called.__setitem__("autosave", called["autosave"] + 1))

    render.set_sink(None)
    try:
        _run(app.run_turn("go"))
    finally:
        render.set_sink(None)

    assert called == {"footer": 0, "todos": 0, "autosave": 1}


# --------------------------------------------------------------------------- #
# 4. Hard cancel (asyncio.CancelledError mid-stream) still finalises state
# --------------------------------------------------------------------------- #

def test_hard_cancel_finalises_state_and_injects_marker(monkeypatch):
    app = _make_session_app()
    pending_ai = AIMessage(
        content="",
        tool_calls=[{"name": "shell", "args": {}, "id": "call-1", "type": "tool_call"}],
    )
    state_messages = [
        HumanMessage(content="run"),
        pending_ai,
        ToolMessage(tool_call_id="call-1", name="shell", content="done"),
    ]

    class _RaisingApp(_FakeApp):
        async def astream_events(self, payload, *, config=None, version="v2"):
            yield {"event": "on_chat_model_start", "name": "agent"}
            raise asyncio.CancelledError()

    fake = _RaisingApp(events=[], state_messages=state_messages, token=app.cancel_token)
    monkeypatch.setattr(app, "app", fake)

    render.set_sink(None)
    try:
        with pytest.raises(asyncio.CancelledError):
            _run(app.run_turn("run"))
    finally:
        render.set_sink(None)

    assert len(fake.updates) == 1
    injected = fake.updates[0]["messages"]
    assert isinstance(injected[-1], AIMessage)
    assert "interrupted" in str(injected[-1].content).lower()


# --------------------------------------------------------------------------- #
# 5. toggle_mode clears/sets pending_act_checkpoint correctly
# --------------------------------------------------------------------------- #

def test_toggle_mode_to_plan_clears_pending_act_checkpoint():
    app = _make_session_app()
    app.agent_mode = "act"
    app._pending_act_checkpoint = True

    app.toggle_mode()

    assert app.agent_mode == "plan"
    assert app._pending_act_checkpoint is False


def test_toggle_mode_to_act_sets_pending_act_checkpoint():
    app = _make_session_app()
    app.agent_mode = "plan"

    app.toggle_mode()

    assert app.agent_mode == "act"
    assert app._pending_act_checkpoint is True


# --------------------------------------------------------------------------- #
# 6. Cancel during the plan->act checkpoint prompt consumes the toggle
# --------------------------------------------------------------------------- #

def test_cancel_during_plan_to_act_checkpoint_consumes_toggle(monkeypatch):
    app = _make_session_app()
    app.agent_mode = "act"
    app._pending_act_checkpoint = True

    state_messages = [HumanMessage(content="run")]

    async def _raising_checkpoint():
        raise asyncio.CancelledError()

    monkeypatch.setattr(app, "_maybe_checkpoint_before_act", _raising_checkpoint)

    fake = _FakeApp(events=[], state_messages=state_messages, token=app.cancel_token)
    monkeypatch.setattr(app, "app", fake)

    render.set_sink(None)
    try:
        with pytest.raises(asyncio.CancelledError):
            _run(app.run_turn("do"))
    finally:
        render.set_sink(None)

    assert app._pending_act_checkpoint is False
    assert len(fake.updates) == 1
    injected = fake.updates[0]["messages"]
    assert isinstance(injected[-1], AIMessage)
    assert "interrupted" in str(injected[-1].content).lower()
