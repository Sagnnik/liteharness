from __future__ import annotations

from liteharness_cli.tui.commands import BUSY_SAFE_COMMANDS


def test_queue_helpers_round_trip(make_app):
    app = make_app()
    assert app.prompt_queue == []
    assert app.dequeue_prompt() is None
    app.enqueue_prompt("first")
    app.enqueue_prompt("second")
    assert len(app.prompt_queue) == 2
    assert app.dequeue_prompt() == "first"
    assert app.dequeue_prompt() == "second"
    assert app.dequeue_prompt() is None


def test_busy_safe_commands_includes_status_and_help():
    assert "status" in BUSY_SAFE_COMMANDS
    assert "help" in BUSY_SAFE_COMMANDS
    assert "config" not in BUSY_SAFE_COMMANDS
    assert "new" not in BUSY_SAFE_COMMANDS
    assert "exit" not in BUSY_SAFE_COMMANDS
