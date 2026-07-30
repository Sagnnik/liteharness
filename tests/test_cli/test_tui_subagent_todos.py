from __future__ import annotations

import asyncio

from liteharness.types import SessionEvent
from liteharness_cli.tui import render


def _transcript_text(app) -> str:
    return "\n".join(line.text for line in app._lines)


def test_mid_turn_todo_refreshes_todos_panel(make_app) -> None:
    app = make_app()
    app.coding.set_todos(
        [{"id": "1", "content": "Ship feature", "status": "in_progress"}]
    )
    app.coding.queue_events(
        SessionEvent("tool_end", {"name": "todo", "content": "Updated 1 todos"}),
        SessionEvent("assistant_final", {"content": "working"}),
    )
    render.set_sink(app)
    try:
        asyncio.run(app._run_turn("do it", []))
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert "todos" in text
    assert "Ship feature" in text
    assert "in_progress" in text


def test_resume_replay_renders_subagent_panel(make_app) -> None:
    app = make_app()
    render.set_sink(app)
    try:
        app._replay_events_to_transcript(
            [
                {
                    "kind": "tool",
                    "tool": "spawn_subagent",
                    "result": "Found routes in src/api.py",
                }
            ]
        )
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert "subagent ok" in text
    assert "Found routes in src/api.py" in text
