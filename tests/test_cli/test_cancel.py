from __future__ import annotations

import asyncio
import os
from types import SimpleNamespace

from cli import render
from cli.session_app import CancelToken, SessionApp


class _FakeApp:
    def __init__(self, events: list[dict], *, trigger_token_after: int | None = None, token: CancelToken | None = None) -> None:
        self._events = list(events)
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
        return type("Snapshot", (), {"values": {"messages": []}})()

    async def aupdate_state(self, config, updates):
        self.updates.append(updates)


def _make_session_app() -> SessionApp:
    os.environ.setdefault("OPENROUTER_API_KEY", "test")
    return SessionApp(git_available=False)


def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_cancel_token_is_set_reset_and_wait():
    token = CancelToken()
    assert token.is_set() is False
    token.trigger()
    assert token.is_set() is True
    token.reset()
    assert token.is_set() is False
    token.trigger()
    asyncio.run(token.wait())


def test_cancel_mid_stream_records_interrupted_text(monkeypatch):
    app = _make_session_app()
    events = [
        {"event": "on_chat_model_start", "name": "agent"},
        {
            "event": "on_chat_model_stream",
            "name": "agent",
            "data": {"chunk": SimpleNamespace(content="Partial assistant text")},
        },
    ]
    fake = _FakeApp(events, trigger_token_after=2, token=app.cancel_token)
    monkeypatch.setattr(app, "app", fake)

    render.set_sink(None)
    try:
        _run(app.run_turn("do something"))
    finally:
        render.set_sink(None)

    assert app.assistant_history
    assert app.assistant_history[-1].endswith("[interrupted]")
    assert "Partial assistant text" in app.assistant_history[-1]
