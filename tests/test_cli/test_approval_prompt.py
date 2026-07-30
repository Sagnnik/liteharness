from __future__ import annotations

from liteharness_cli.tui.models import MenuItem
from liteharness_cli.tui.models import TranscriptLine


def test_approval_command_is_rendered_above_choices_in_green(make_app) -> None:
    app = make_app()
    app._menu_kind = "approval"
    app._prompt_summary_lines = ["pytest -q"]
    app._prompt_items = [MenuItem("yes", "approve once")]

    fragments = app._menu_body_fragments()

    command_index = next(
        index
        for index, (style, text) in enumerate(fragments)
        if "pytest -q" in text and style == "class:chrome.approval.command"
    )
    choice_index = next(
        index
        for index, (_, text) in enumerate(fragments)
        if "approve once" in text
    )
    assert command_index < choice_index


def test_clear_transcript_removes_backing_lines(make_app) -> None:
    app = make_app()
    app._transcript_store.append([TranscriptLine("", "old session")])

    app.clear_transcript()

    assert app._transcript_store.lines == []
    assert "old session" not in app._transcript_store.plain_text()
