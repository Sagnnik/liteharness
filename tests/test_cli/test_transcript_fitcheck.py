"""Regression tests for the incremental user-band fit check.

``_after_render`` runs after every prompt_toolkit frame. The user-band width
validation must stay O(new lines) instead of O(transcript) — a full rescan
per frame made the spinner/streaming repaint stall on long transcripts.
The reflow behaviour itself (misfit detection, resize rebuild) is unchanged.
"""

from __future__ import annotations

from ness_cli.tui.formatting import USER_STYLE, user_band_width, user_message_lines


def _prime(app, width: int = 120, height: int = 40) -> None:
    """Set a render width, append one user block, and validate it."""
    app._on_transcript_render_size(width, height)
    app.append_user("first message")
    app._after_render()


def _user_rows(app):
    return [line for line in app._lines if line.style == USER_STYLE]


def test_after_render_scans_only_new_tail(make_app, monkeypatch):
    app = make_app()
    _prime(app)
    # Grow the transcript with non-user lines and validate them too.
    app.append_assistant("some **markdown** output\n" * 20)
    app._after_render()

    calls: list[int] = []
    real = app._user_blocks_fit_width

    def spy(width, *, start=0):
        calls.append(start)
        return real(width, start=start)

    monkeypatch.setattr(app, "_user_blocks_fit_width", spy)
    app._after_render()
    # Steady-state frame: nothing new appended, so the scan starts (and ends)
    # at the buffer tail instead of rescanning from line 0.
    assert calls == [len(app._lines)]


def test_after_render_detects_misfit_in_new_tail(make_app):
    app = make_app()
    width = 120
    _prime(app, width)
    # A stale user block (built at a narrower width) lands after the
    # already-validated prefix; the next frame must detect and rebuild it.
    app._append_transcript(*user_message_lines("stale", width=width - 40))
    app._after_render()
    expected = user_band_width(width=width)
    assert _user_rows(app)
    for line in _user_rows(app):
        assert len(line.text) == expected


def test_width_change_still_reflows(make_app):
    app = make_app()
    _prime(app, width=120)
    app._on_transcript_render_size(80, 40)
    app._after_render()
    expected = user_band_width(width=80)
    for line in _user_rows(app):
        assert len(line.text) == expected
    # The full rebuild validates the whole buffer at the new width.
    assert app._user_fit_checked_upto == len(app._lines)
    assert app._layout_term_width == 80


def test_fit_check_backs_up_to_block_start(make_app):
    app = make_app()
    _prime(app)
    block_start = next(i for i, line in enumerate(app._lines) if line.user_source)
    assert app._lines[block_start + 1].style == USER_STYLE
    app._lines[block_start + 1].text = "short"
    # A scan starting mid-block must walk back to the block's first row and
    # still catch the misfit.
    assert not app._user_blocks_fit_width(120, start=block_start + 1)


def test_clear_transcript_resets_fit_cursor(make_app):
    app = make_app()
    _prime(app)
    app.clear_transcript()
    assert app._user_fit_checked_upto == 0
