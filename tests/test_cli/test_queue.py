from __future__ import annotations

from cli.commands import BUSY_SAFE_COMMANDS
from tests.test_cli.helpers import make_app


def test_queue_helpers_round_trip():
    app = make_app()
    sess = app.session
    assert sess.prompt_queue == []
    assert sess.dequeue_prompt() is None
    sess.enqueue_prompt("first")
    sess.enqueue_prompt("second")
    assert len(sess.prompt_queue) == 2
    assert sess.dequeue_prompt() == "first"
    assert sess.dequeue_prompt() == "second"
    assert sess.dequeue_prompt() is None


def test_busy_safe_commands_includes_status_and_help():
    assert "status" in BUSY_SAFE_COMMANDS
    assert "help" in BUSY_SAFE_COMMANDS
    assert "config" not in BUSY_SAFE_COMMANDS
    assert "reset" not in BUSY_SAFE_COMMANDS
    assert "exit" not in BUSY_SAFE_COMMANDS
