from __future__ import annotations

import asyncio
from datetime import datetime

from ness_cli.tui.models import MenuItem
from ness_cli.tui.prompts import format_thread_updated_at


def test_scrollable_picker_expands_to_twelve_rows(make_app, monkeypatch) -> None:
    app = make_app()
    monkeypatch.setattr("ness_cli.tui.pickers.term_height", lambda: 40)
    app._menu_kind = "threads"
    app._prompt_kind = "threads"
    app._prompt_items = [MenuItem(str(i), f"thread {i}") for i in range(20)]

    option_rows, _, _ = app._menu_layout_rows()
    assert option_rows == 12
    assert app._menu_body_height() == 12

    app._menu_index = 15
    app._clamp_menu_scroll()
    assert app._menu_scroll == 4


def test_short_terminal_reserves_transcript_rows_for_question(make_app, monkeypatch) -> None:
    app = make_app()
    monkeypatch.setattr("ness_cli.tui.pickers.term_height", lambda: 24)
    monkeypatch.setattr("ness_cli.tui.transcript.term_height", lambda: 24)
    app._menu_kind = "question"
    app._prompt_kind = "question"
    app._prompt_title = "question 1: choose"
    app._prompt_question = {"allow_note": True}
    app._prompt_items = [MenuItem(str(i), f"option {i}") for i in range(20)]

    assert app._menu_layout_rows()[0] == 10
    assert app._chrome_height_lines() == 18
    assert app._transcript_viewport_lines() == 6


def test_approval_details_shrink_choices_before_chrome_overflows(
    make_app, monkeypatch
) -> None:
    app = make_app()
    monkeypatch.setattr("ness_cli.tui.pickers.term_height", lambda: 24)
    app._menu_kind = "approval"
    app._prompt_kind = "approval"
    app._prompt_items = [MenuItem(str(i), f"choice {i}") for i in range(7)]
    app._prompt_summary_lines = ["one", "two", "three"]

    assert app._menu_layout_rows()[0] == 8
    assert app._menu_body_height() == 11

    app._prompt_detail_lines = [f"detail {i}" for i in range(8)]
    option_rows, _, detail_lines = app._menu_layout_rows()
    assert option_rows == 3
    assert len(detail_lines) == 4
    assert app._menu_body_height() == 12


def test_question_note_visibility_honors_allow_note(make_app) -> None:
    app = make_app()
    app._prompt_kind = "question"
    app._prompt_question = {"allow_note": False}
    assert app._form_visible() is False

    app._prompt_question = {"allow_note": True}
    assert app._form_visible() is True


def test_picker_header_stays_within_terminal_width(make_app, monkeypatch) -> None:
    app = make_app()
    monkeypatch.setattr("ness_cli.tui.pickers.term_width", lambda: 40)
    app._menu_kind = "question"
    app._prompt_title = "question 1: " + ("very long prompt " * 8)
    app._prompt_hint = "↑/↓ option · Tab note · Enter submit · Esc cancel"

    fragments = app._menu_header_fragments()
    assert sum(len(text) for _, text in fragments) == 40
    assert any("…" in text for _, text in fragments)


def test_thread_picker_prefixes_local_updated_datetime(make_app) -> None:
    app = make_app()
    updated_at = "2026-08-12T14:37:00+00:00"
    expected = datetime.fromisoformat(updated_at).astimezone().strftime("%Y-%m-%d %H:%M")

    async def exercise() -> str:
        task = asyncio.create_task(
            app.request_threads_picker(
                [
                    {
                        "thread_id": "session-one",
                        "name": "Release prep",
                        "updated_at": updated_at,
                    }
                ],
                current_thread_id="session-other",
            )
        )
        await asyncio.sleep(0)
        label = app._prompt_items[0].label
        app._cancel_menu()
        await task
        return label

    assert asyncio.run(exercise()) == f"{expected}  Release prep"
    assert format_thread_updated_at("not-a-date") == ""


def test_thread_picker_marks_live_turns_as_working(make_app) -> None:
    app = make_app()

    async def exercise() -> str:
        task = asyncio.create_task(
            app.request_threads_picker(
                [
                    {
                        "thread_id": "session-working",
                        "name": "Background task",
                        "live_status": {"status": "working", "elapsed": 2.0},
                    }
                ],
                current_thread_id="session-other",
            )
        )
        await asyncio.sleep(0)
        suffix = app._prompt_items[0].suffix
        app._cancel_menu()
        await task
        return suffix

    suffix = asyncio.run(exercise())
    assert "working" in suffix
    assert suffix.startswith("⠋")


def test_thread_picker_clears_spinner_when_turn_finishes(make_app) -> None:
    app = make_app()

    async def exercise() -> str:
        release = asyncio.Event()

        async def working() -> None:
            await release.wait()

        runtime = app._runtime()
        runtime.task = asyncio.create_task(working())
        runtime.status = "working"
        runtime.started_at = 0.0
        picker = asyncio.create_task(
            app.request_threads_picker(
                [
                    {
                        "thread_id": app.thread_id,
                        "name": "Finishing task",
                        "live_status": {"status": "working", "elapsed": 1.0},
                    }
                ],
                current_thread_id=app.thread_id,
            )
        )
        await asyncio.sleep(0)
        release.set()
        await runtime.task
        await asyncio.sleep(0.1)
        suffix = app._prompt_items[0].suffix
        app._cancel_menu()
        await picker
        return suffix

    assert "working" not in asyncio.run(exercise())
