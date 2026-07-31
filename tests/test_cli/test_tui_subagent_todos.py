from __future__ import annotations

import asyncio

from ness_ai.types import SessionEvent
from ness_cli.tui import render


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


def test_todo_block_moves_to_end_on_refresh(make_app) -> None:
    app = make_app()
    render.set_sink(app)
    try:
        render.render_todos(
            [{"id": "1", "content": "First item", "status": "in_progress"}]
        )
        render.render_user_echo("later user turn")
        render.render_assistant_panel("later assistant reply")
        render.render_todos(
            [{"id": "1", "content": "First item", "status": "in_progress"}]
        )
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert text.index("later user turn") < text.rindex("First item")
    assert text.index("later assistant reply") < text.rindex("First item")
    assert text.rstrip().endswith("First item") or "First item" in text.splitlines()[-5:]


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


def test_resume_replay_renders_edit_diff_and_shell_panel(make_app) -> None:
    app = make_app()
    render.set_sink(app)
    try:
        app._replay_events_to_transcript(
            [
                {
                    "kind": "tool",
                    "tool": "edit",
                    "result": "Updated foo.py\ndiff:\n@@ -1 +1 @@\n-old\n+new\n",
                    "exit": "ok",
                },
                {
                    "kind": "tool",
                    "tool": "shell",
                    "result": "status=ok\nexit_code=0\noutput:\nhello from shell\n",
                    "exit": "ok",
                },
                {
                    "kind": "tool",
                    "tool": "read",
                    "result": "Denied by user approval: shell",
                    "exit": "denied",
                },
            ]
        )
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert "diff edit" in text
    assert "+new" in text
    assert "shell status=ok" in text
    assert "hello from shell" in text
    assert "[denied]" in text


def test_resume_replay_denied_shell_and_subagent_show_exit_status(make_app) -> None:
    app = make_app()
    render.set_sink(app)
    try:
        app._replay_events_to_transcript(
            [
                {
                    "kind": "tool",
                    "tool": "shell",
                    "result": "Denied by user approval: shell",
                    "exit": "denied",
                },
                {
                    "kind": "tool",
                    "tool": "spawn_subagent",
                    "result": "Denied by user approval: spawn_subagent",
                    "exit": "denied",
                },
            ]
        )
    finally:
        render.set_sink(None)

    text = _transcript_text(app)
    assert "[denied]" in text
    assert "Denied by user approval: shell" in text
    assert "Denied by user approval: spawn_subagent" in text
    assert "shell status=ok" not in text
    assert "subagent ok" not in text
