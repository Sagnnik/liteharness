"""TUI-side interrupted-turn rendering.

The SDK owns checkpoint cleanup on cancel (covered by the SDK/adapter
suites); these tests pin the TUI seam: an ``interrupted`` SessionEvent
mid-turn stops the live stream, records the partial assistant text with the
interrupted suffix for /copy, renders the cancel banner, and suppresses the
normal usage/todos footer.
"""

from __future__ import annotations

import asyncio

from liteharness.types import SessionEvent

from cli import render


def _run_turn(app, text: str = "do something") -> None:
    render.set_sink(app)
    try:
        asyncio.run(app._run_turn(text, []))
    finally:
        render.set_sink(None)


def _transcript_text(app) -> str:
    return "\n".join(line.text for line in app._lines)


def test_cancel_mid_stream_records_interrupted_text(make_app):
    app = make_app()
    app.coding.queue_events(
        SessionEvent("assistant_delta", {"text": "Partial assistant text"}),
        SessionEvent("interrupted", {"partial_text": "Partial assistant text"}),
    )
    _run_turn(app)

    assert app.assistant_history
    assert app.assistant_history[-1].endswith("[interrupted]")
    assert "Partial assistant text" in app.assistant_history[-1]


def test_cancel_renders_banner_and_suppresses_footer(make_app, monkeypatch):
    app = make_app()
    footers: list[dict] = []
    monkeypatch.setattr(
        "cli.app.render.render_usage_footer", lambda usage: footers.append(usage)
    )
    app.coding.queue_events(
        SessionEvent("assistant_delta", {"text": "half"}),
        SessionEvent("usage", {"model": "m", "input_tokens": 10, "output_tokens": 5}),
        SessionEvent("interrupted", {"partial_text": "half"}),
    )
    _run_turn(app)

    assert "Turn interrupted by user." in _transcript_text(app)
    assert footers == []


def test_normal_turn_records_text_and_renders_footer(make_app, monkeypatch):
    app = make_app()
    footers: list[dict] = []
    monkeypatch.setattr(
        "cli.app.render.render_usage_footer", lambda usage: footers.append(usage)
    )
    app.coding.queue_events(
        SessionEvent("assistant_delta", {"text": "hello "}),
        SessionEvent("assistant_delta", {"text": "world"}),
        SessionEvent("assistant_final", {"content": "hello world"}),
        SessionEvent("usage", {"model": "m", "input_tokens": 12, "output_tokens": 3}),
    )
    _run_turn(app, "hi")

    assert app.assistant_history[-1] == "hello world"
    assert app.coding.turn_count == 1
    assert footers == [
        {
            "model": "m",
            "input_tokens": 12,
            "uncached_input_tokens": 0,
            "cached_input_tokens": 0,
            "output_tokens": 3,
        }
    ]


def test_non_streamed_final_renders_panel(make_app):
    """assistant_final with no preceding deltas (non-streaming model) still
    lands in the transcript via the panel path."""
    app = make_app()  # FakeCoding defaults to a final-only echo
    _run_turn(app, "ping")

    assert app.assistant_history[-1] == "echo ping"
    assert "echo ping" in _transcript_text(app)
