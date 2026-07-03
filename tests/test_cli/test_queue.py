from __future__ import annotations

import asyncio

from cli import render
from cli.commands import BUSY_SAFE_COMMANDS, dispatch
from tests.test_cli.helpers import make_app


async def _dispatch_with_sink(app, command, *, busy=False) -> None:
    render.set_sink(app)
    try:
        await dispatch(app.session, command, busy=busy)
    finally:
        render.set_sink(None)


def _transcript_text(app) -> str:
    return "\n".join(line.text for line in app._lines)


# --------------------------------------------------------------------------- #
# Queue primitives
# --------------------------------------------------------------------------- #

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


def test_queued_prompt_compat_property_sets_single_item():
    app = make_app()
    sess = app.session
    sess.queued_prompt = "from-disk"
    assert sess.prompt_queue == ["from-disk"]
    assert sess.queued_prompt == "from-disk"
    sess.queued_prompt = ""
    assert sess.prompt_queue == []


# --------------------------------------------------------------------------- #
# Busy-safe command dispatch
# --------------------------------------------------------------------------- #

def test_busy_safe_commands_includes_status_and_help():
    assert "status" in BUSY_SAFE_COMMANDS
    assert "help" in BUSY_SAFE_COMMANDS
    assert "config" not in BUSY_SAFE_COMMANDS
    assert "reset" not in BUSY_SAFE_COMMANDS
    assert "exit" not in BUSY_SAFE_COMMANDS


def test_safe_slash_runs_while_busy():
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/status", busy=True))
    text = _transcript_text(app)
    assert "session status" in text


def test_unsafe_slash_rejected_while_busy():
    app = make_app()
    asyncio.run(_dispatch_with_sink(app, "/config", busy=True))
    text = _transcript_text(app)
    assert "not available while a task is running" in text
    assert app.session.rebuilt is False


# --------------------------------------------------------------------------- #
# Schedule submit — enqueue when busy, start turn when not
# --------------------------------------------------------------------------- #

def test_schedule_submit_enqueues_plain_text_while_busy():
    app = make_app()
    app._busy = True
    app._schedule_submit("next prompt")
    assert app.session.prompt_queue == ["next prompt"]
    text = _transcript_text(app)
    assert "[queue]" in text
    assert "next prompt" in text
    assert app._buffer.text == ""


def test_schedule_submit_starts_turn_when_not_busy():
    app = make_app()
    started: list[str] = []

    async def fake_run_turn(text):
        started.append(text)

    app.session.run_turn = fake_run_turn

    async def _run():
        app._app.create_background_task = lambda coro: asyncio.ensure_future(coro)
        app._schedule_submit("hello")
        for _ in range(50):
            if started:
                return
            await asyncio.sleep(0)

    asyncio.run(_run())
    assert started == ["hello"]
    assert app.session.prompt_queue == []


# --------------------------------------------------------------------------- #
# Queue drain and cancel
# --------------------------------------------------------------------------- #

def test_drain_queue_after_turn_runs_in_order():
    app = make_app()
    from cli.commands import dispatch as real_dispatch

    seen: list[str] = []

    async def fake_run_turn(text):
        seen.append(text)

    app.session.run_turn = fake_run_turn
    app._dispatch = real_dispatch
    app.session.prompt_queue = ["q1", "q2"]

    asyncio.run(app._submit_async("/help"))
    assert "q1" in seen
    assert "q2" in seen
    assert seen.index("q1") < seen.index("q2")


def test_cancel_active_task_clears_queue():
    app = make_app()
    app.session.prompt_queue = ["pending1", "pending2"]

    async def never_finish(text):
        await asyncio.sleep(10)

    app.session.run_turn = never_finish

    async def _run():
        app._app.create_background_task = lambda coro: asyncio.ensure_future(coro)
        app._schedule_submit("running")
        cancelled = app._cancel_active_task()
        assert cancelled is True
        assert app.session.prompt_queue == []
        text = _transcript_text(app)
        assert "cleared 2 queued prompt" in text

    asyncio.run(_run())
