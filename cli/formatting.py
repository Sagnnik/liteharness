from __future__ import annotations

import textwrap

from cli.theme import GRAY, GRAY_BRIGHT, GRAY_DIM
from cli.models import TranscriptLine
from cli.utils import term_width

_SPINNER_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
# Deep gray base with a light gray / near-white band that sweeps across the label.
_WORKING_COLORS = (GRAY_DIM, GRAY_DIM, GRAY, GRAY_BRIGHT, "#f0f2f5", GRAY_BRIGHT, GRAY, GRAY_DIM)

USER_STYLE = "class:transcript.user"
_USER_STYLE = USER_STYLE

_USER_H_PAD = 2
_USER_V_PAD = 1


def user_band_width(*, width: int | None = None) -> int:
    """Full-width band row length (render width minus BufferControl trailing space)."""
    render_width = width if width is not None else term_width()
    return max(20, render_width - 1)


def _band_row(content: str, width: int) -> TranscriptLine:
    body = content[:width]
    tail = max(0, width - len(body))
    line = body + (" " * tail)
    return TranscriptLine(_USER_STYLE, line, fragments=[(_USER_STYLE, line)])


def _wrap_user_text(text: str, width: int) -> list[str]:
    paragraphs = str(text).splitlines() or [""]
    lines: list[str] = []
    for paragraph in paragraphs:
        if not paragraph.strip():
            lines.append("")
            continue
        wrapped = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=True,
            break_on_hyphens=False,
        )
        lines.extend(wrapped or [""])
    return lines or [""]


def user_message_lines(text: str, *, width: int | None = None) -> list[TranscriptLine]:
    """Full-width user block: pad rows + text rows, one shared background (Pi-style band)."""
    stripped = text.strip()
    if not stripped:
        return []

    band_width = user_band_width(width=width)
    wrap_width = max(8, band_width - _USER_H_PAD)
    body_lines = _wrap_user_text(stripped, wrap_width)

    rows: list[TranscriptLine] = []
    for _ in range(_USER_V_PAD):
        rows.append(_band_row("", band_width))
    for line in body_lines:
        rows.append(_band_row((" " * _USER_H_PAD) + line, band_width))
    for _ in range(_USER_V_PAD):
        rows.append(_band_row("", band_width))
    if rows:
        rows[0].user_source = stripped
    return rows


def working_fragments(frame: int) -> list[tuple[str, str]]:
    spinner = _SPINNER_FRAMES[frame % len(_SPINNER_FRAMES)]
    word = "Working"
    fragments: list[tuple[str, str]] = [("class:chrome.working.spinner", f"{spinner} ")]
    for index, char in enumerate(word):
        color = _WORKING_COLORS[(frame + index) % len(_WORKING_COLORS)]
        fragments.append((color, char))
    return fragments


def worked_fragments(elapsed_s: float) -> list[tuple[str, str]]:
    label = f"Worked for {_format_duration(elapsed_s)}"
    return [("class:chrome.worked", label)]


def _format_duration(elapsed_s: float) -> str:
    if elapsed_s < 60:
        return f"{elapsed_s:.1f}s"
    minutes = int(elapsed_s // 60)
    seconds = elapsed_s % 60
    return f"{minutes}m {seconds:.1f}s"
