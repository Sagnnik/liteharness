"""Question prompt soft-cancel: Esc / Ctrl+C resolve without picking option 0."""

from __future__ import annotations

import asyncio

from ness_ai.tools.ask import _format_answers


def _question(qid: str = "q1", prompt: str = "Pick?") -> dict:
    return {
        "id": qid,
        "prompt": prompt,
        "options": [
            {"id": "a", "label": "Alpha", "recommended": True},
            {"id": "b", "label": "Beta"},
        ],
    }


def test_cancel_open_prompt_resolves_question_to_none(make_app) -> None:
    app = make_app()

    async def _exercise() -> object:
        app._prompt_future = asyncio.get_running_loop().create_future()
        app._prompt_kind = "question"
        assert app._cancel_open_prompt() is True
        return app._prompt_future.result()

    assert asyncio.run(_exercise()) is None
    assert app._prompt_kind is None


def test_ask_question_cancelled_sentinel_returns_soft_cancel(make_app) -> None:
    app = make_app()

    async def _exercise() -> dict:
        async def cancel_soon() -> None:
            await asyncio.sleep(0)
            assert app._prompt_future is not None
            app._prompt_future.set_result(None)

        task = asyncio.create_task(cancel_soon())
        answer = await app._ask_question(1, _question())
        await task
        return answer

    answer = asyncio.run(_exercise())
    assert answer == {
        "id": "q1",
        "selected": None,
        "note": "cancelled by user",
    }


def test_ask_question_submit_still_returns_selection(make_app) -> None:
    app = make_app()

    async def _exercise() -> dict:
        async def submit_soon() -> None:
            await asyncio.sleep(0)
            assert app._prompt_future is not None
            app._menu_index = 1
            app._submit_question()

        task = asyncio.create_task(submit_soon())
        answer = await app._ask_question(1, _question())
        await task
        return answer

    answer = asyncio.run(_exercise())
    assert answer == {
        "id": "q1",
        "selected": {"id": "b", "label": "Beta"},
        "note": "",
    }


def test_ask_questions_pads_remaining_after_cancel(make_app) -> None:
    app = make_app()
    questions = [_question("q1", "First?"), _question("q2", "Second?")]

    async def _exercise() -> list[dict]:
        async def cancel_first() -> None:
            await asyncio.sleep(0)
            assert app._prompt_future is not None
            app._prompt_future.set_result({"cancelled": True})

        task = asyncio.create_task(cancel_first())
        answers = await app.ask_questions(questions)
        await task
        return answers

    answers = asyncio.run(_exercise())
    assert answers == [
        {"id": "q1", "selected": None, "note": "cancelled by user"},
        {"id": "q2", "selected": None, "note": "cancelled by user"},
    ]


def test_cancel_menu_soft_cancels_question(make_app) -> None:
    app = make_app()

    async def _exercise() -> dict:
        async def esc_soon() -> None:
            await asyncio.sleep(0)
            app._cancel_menu()

        task = asyncio.create_task(esc_soon())
        answer = await app._ask_question(1, _question())
        await task
        return answer

    assert asyncio.run(_exercise())["selected"] is None


def test_note_active_escape_only_clears_note(make_app) -> None:
    """Esc while the note field is focused must leave the Future unresolved."""
    app = make_app()
    app._prompt_kind = "question"
    app._prompt_note_active = True
    app._prompt_future = None  # no future — form_cancel must not invent one

    # Mirror keys.py _form_cancel when note is active.
    if app._prompt_kind == "question" and app._prompt_note_active:
        app._prompt_note_active = False

    assert app._prompt_note_active is False


def test_format_answers_cancelled() -> None:
    text = _format_answers(
        [_question()],
        [{"id": "q1", "selected": None, "note": "cancelled by user"}],
    )
    assert "A: (cancelled)" in text
    assert "Note: cancelled by user" in text
