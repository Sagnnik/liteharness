from __future__ import annotations

import pytest

from ness_cli.tui import render
from ness_cli.tui.models import TranscriptLine
from ness_cli.tui.widgets import TranscriptStore


def _line(text: str) -> TranscriptLine:
    return TranscriptLine("", text)


def _text(app) -> str:
    return "\n".join(line.text for line in app._lines)


def test_table_wraps_cells_to_transcript_width(make_app) -> None:
    app = make_app()
    app._on_transcript_render_size(48, 20)

    app.append_table(
        "skills (1)",
        ["skill", "source", "description"],
        [["example-skill", ".agents/skills/example/SKILL.md", "description " * 12]],
    )

    assert max(len(line.text) for line in app._lines) <= 48
    assert "description" in _text(app)
    assert "─" in _text(app)


def test_tracked_blocks_follow_structural_mutations() -> None:
    store = TranscriptStore([_line("prefix")])
    first = store.append_tracked([_line("first")])
    second = store.append_tracked([_line("second")])

    store.insert(0, [_line("before")])
    assert (first.start, second.start) == (2, 3)

    store.replace_tracked(first, [_line("first-a"), _line("first-b")])
    assert (first.start, first.count, second.start) == (2, 2, 4)

    store.delete_tracked(first)
    assert not first.attached
    assert second.start == 2
    assert [line.text for line in store.lines] == ["before", "prefix", "second"]

    store.release_tracked(second)
    assert not second.attached
    assert store.lines[2].text == "second"


def test_raw_mutations_cannot_overlap_or_split_tracked_blocks() -> None:
    store = TranscriptStore([])
    block = store.append_tracked([_line("a"), _line("b")])

    with pytest.raises(ValueError, match="overlaps a tracked block"):
        store.replace(block.start, block.count, [_line("replacement")])
    with pytest.raises(ValueError, match="splits a tracked block"):
        store.insert(block.start + 1, [_line("middle")])


def test_reset_detaches_all_tracked_blocks() -> None:
    store = TranscriptStore([])
    block = store.append_tracked([_line("answer")])
    store.reset([])
    assert not block.attached
    assert store.lines == []


@pytest.mark.parametrize("feed_after_toggle", [False, True])
def test_reasoning_toggle_keeps_live_answer_block_and_final_markdown(
    make_app, feed_after_toggle: bool
) -> None:
    app = make_app()
    app._on_transcript_render_size(40, 20)
    app.append_reasoning("old reasoning " * 30, elapsed=1.0)

    stream = app.start_assistant_stream()
    stream.feed("new **answer**")
    assert stream.block is not None
    old_start = stream.block.start

    app.toggle_reasoning()
    assert stream.block.start > old_start
    if feed_after_toggle:
        stream.feed("\nsecond line")
    stream.stop()

    text = _text(app)
    assert text.count("new answer") == 1
    assert "new **answer**" not in text
    if feed_after_toggle:
        assert text.count("second line") == 1
    assert not stream.block.attached


def test_repeated_reasoning_toggles_during_stream_do_not_duplicate_answer(
    make_app,
) -> None:
    app = make_app()
    app._on_transcript_render_size(40, 20)
    app.append_reasoning("old reasoning " * 30, elapsed=1.0)
    stream = app.start_assistant_stream()
    stream.feed("stable **answer**")

    app.toggle_reasoning()
    app.toggle_reasoning()
    app.toggle_reasoning()
    stream.feed("\ncontinued")
    stream.stop()

    text = _text(app)
    assert text.count("stable answer") == 1
    assert "stable **answer**" not in text
    assert text.count("continued") == 1


@pytest.mark.parametrize("reasoning_first", [False, True])
def test_reasoning_slot_and_stream_follow_actual_facade_order(
    make_app, reasoning_first: bool
) -> None:
    app = make_app()
    app._on_transcript_render_size(40, 20)
    app.append_reasoning("older reasoning " * 20, elapsed=1.0)
    render.set_sink(app)
    try:
        stream = render.AssistantStream()
        if reasoning_first:
            stream.feed_reasoning("current reasoning " * 20)
            stream.feed("facade **answer**")
        else:
            stream.feed("facade **answer**")
            stream.feed_reasoning("current reasoning " * 20)
        app.toggle_reasoning()
        stream.feed("\ncontinued")
        stream.stop()
        stream.finalize_reasoning()
    finally:
        render.set_sink(None)

    text = _text(app)
    assert text.count("facade answer") == 1
    assert "facade **answer**" not in text
    assert text.count("continued") == 1
    assert "current reasoning" in text


def test_user_reflow_shifts_live_stream_handle(make_app) -> None:
    app = make_app()
    app._on_transcript_render_size(50, 20)
    app.append_user("wrapped user message " * 30)
    app.append_reasoning("historical reasoning " * 20, elapsed=1.0)
    stream = app.start_assistant_stream()
    stream.feed("resize **answer**")

    app._on_transcript_render_size(100, 20)
    app._after_render()
    assert stream.block is not None
    assert app._lines[stream.block.start].text == "resize **answer**"

    app.toggle_reasoning()
    stream.stop()
    text = _text(app)
    assert text.count("resize answer") == 1
    assert "resize **answer**" not in text


def test_todo_move_and_reasoning_toggle_preserve_live_answer(make_app) -> None:
    app = make_app()
    todo = [{"id": "1", "content": "First item", "status": "in_progress"}]
    app.append_todos(todo)
    app.append_reasoning("historical reasoning " * 20, elapsed=1.0)
    stream = app.start_assistant_stream()
    stream.feed("todo-safe **answer**")

    app.append_todos(todo)
    app.toggle_reasoning()
    stream.stop()

    text = _text(app)
    assert text.count("First item") == 1
    assert text.count("todo-safe answer") == 1
    assert "todo-safe **answer**" not in text


def test_turn_finish_moves_last_answer_below_late_tools_after_toggle(make_app) -> None:
    app = make_app()
    app._on_transcript_render_size(40, 20)
    app._turn_render_active = True

    stream = app.start_assistant_stream()
    stream.feed("final **answer**")
    stream.stop()
    app.append_tool_calls(
        [{"name": "first_tool", "args": {}, "id": "1", "type": "tool_call"}]
    )
    app.append_reasoning("later reasoning " * 20, elapsed=1.0)
    app.toggle_reasoning()
    app.append_tool_calls(
        [{"name": "second_tool", "args": {}, "id": "2", "type": "tool_call"}]
    )

    assert _text(app).index("final answer") < _text(app).index("second_tool")
    app._finalize_turn_assistant_order()
    app._turn_render_active = False

    text = _text(app)
    assert text.count("final answer") == 1
    assert text.index("first_tool") < text.index("final answer")
    assert text.index("second_tool") < text.index("final answer")
    assert app._last_assistant_block is None

    # Moving the answer crosses the tracked reasoning span; its handle must
    # remain valid for future Ctrl+T toggles.
    app.toggle_reasoning()
    assert _text(app).count("final answer") == 1
    assert _text(app).index("second_tool") < _text(app).index("final answer")


def test_newer_answer_releases_intermediate_candidate(make_app) -> None:
    app = make_app()
    app._turn_render_active = True

    first = app.start_assistant_stream()
    first.feed("intermediate narration")
    first.stop()
    app.append_tool_calls(
        [{"name": "lookup", "args": {}, "id": "1", "type": "tool_call"}]
    )
    second = app.start_assistant_stream()
    second.feed("actual final")
    second.stop()
    app._finalize_turn_assistant_order()
    app._turn_render_active = False

    text = _text(app)
    assert text.index("intermediate narration") < text.index("lookup")
    assert text.index("lookup") < text.index("actual final")
