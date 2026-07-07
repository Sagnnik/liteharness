from __future__ import annotations

import io
import re
from typing import Any

from rich.console import Console
from rich.markdown import Markdown

from cli.theme import RICH_THEME
from cli.models import TranscriptLine

_STATUS_MARKER: dict[str, str] = {
    "completed": "[x]",
    "in_progress": "[~]",
    "pending": "[ ]",
    "cancelled": "[-]",
}


def todos_transcript_lines(todos: list[dict[str, Any]], *, width: int) -> list[TranscriptLine]:
    del width
    if not todos:
        return []

    status_width = max(len("status"), *(len(str(todo.get("status", ""))) for todo in todos))
    lines: list[TranscriptLine] = [
        TranscriptLine("class:transcript.todo.title", "todos"),
        TranscriptLine(
            "class:transcript.muted",
            f"  {'status'.ljust(status_width)}  task",
        ),
        TranscriptLine("class:transcript.muted", f"  {'-' * status_width}  {'-' * 24}"),
    ]

    for todo in todos:
        status = str(todo.get("status", ""))
        marker = _STATUS_MARKER.get(status, "[ ]")
        content = str(todo.get("content", ""))
        lines.append(
            TranscriptLine(
                "class:transcript.panel",
                f"  {marker} {status.ljust(status_width)}  {content}",
            )
        )
    return lines


# --- markdown via Rich -> prompt_toolkit ANSI bridge ------------------------
# Rich owns the terminal when used standalone, so calling the shared stdout
# console inside the running PTK app would garble/crash it. Instead we render
# Markdown into an in-memory truecolor ANSI string with a dedicated StringIO
# console (never touching stdout) and bridge the result into PTK formatted text
# via prompt_toolkit.formatted_text.ANSI. PTK's ANSI parser cannot decode Rich's
# OSC 8 hyperlink wrappers (the URL leaks as visible text), so those are stripped
# first.

_ESC = chr(27)
_BEL = chr(7)
# OSC 8 hyperlink: ESC ] 8 ; params ; uri (ST | BEL) ... ESC ] 8 ; ; (ST | BEL)
_OSC8_RE = re.compile(
    re.escape(_ESC + "]8;")
    + "[^" + _ESC + _BEL + "]*"
    + "(?:" + re.escape(_ESC + chr(92)) + "|" + re.escape(_BEL) + ")"
)

# Reusable, never-stdout console. Width is set per render so Rich wraps to the
# transcript column and emits full-width code-block backgrounds.
_MD_CONSOLE = Console(
    file=io.StringIO(),
    theme=RICH_THEME,
    force_terminal=True,
    color_system="truecolor",
    legacy_windows=False,
    width=80,
)


def _render_markdown_ansi(text: str, *, width: int) -> str:
    _MD_CONSOLE.width = max(8, width)
    with _MD_CONSOLE.capture() as capture:
        _MD_CONSOLE.print(Markdown(text))
    return _OSC8_RE.sub("", capture.get())


def _ansi_to_rows(ansi: str) -> list[list[tuple[str, str]]]:
    from prompt_toolkit.formatted_text import ANSI, to_formatted_text
    from prompt_toolkit.formatted_text.utils import split_lines

    fragments = [(_frag_style(f), _frag_text(f)) for f in to_formatted_text(ANSI(ansi))]
    rows: list[list[tuple[str, str]]] = []
    for line_frags in split_lines(fragments):
        rows.append(_coalesce([(s, t) for s, t in line_frags]))
    return rows


def _frag_style(frag: Any) -> str:
    return str(frag[0]) if frag[0] else ""


def _frag_text(frag: Any) -> str:
    return str(frag[1]) if len(frag) > 1 else ""


def _coalesce(frags: list[tuple[str, str]]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for style, text in frags:
        if not text:
            continue
        if out and out[-1][0] == style:
            out[-1] = (style, out[-1][1] + text)
        else:
            out.append((style, text))
    return out


def _plain_lines(text: str) -> list[TranscriptLine]:
    return [TranscriptLine("class:transcript.assistant", line) for line in text.split("\n")]


def markdown_transcript_lines(text: str, *, width: int) -> list[TranscriptLine]:
    """Render assistant markdown into styled TranscriptLines using Rich.

    Rich provides the layout (centered headings, code-block backgrounds with
    Pygments highlighting, blockquote bars, list bullets, blank-line spacing
    between blocks). The result is bridged into prompt_toolkit fragments; each
    TranscriptLine's fragments concatenate back to its ``text`` (an invariant
    TranscriptStore relies on at widgets.py:_line_fragments).
    """
    stripped = (text or "").strip()
    if not stripped:
        return [TranscriptLine("class:transcript.assistant", "")]
    try:
        ansi = _render_markdown_ansi(stripped, width=width)
        rows = _ansi_to_rows(ansi)
    except Exception:
        return _plain_lines(stripped)

    lines: list[TranscriptLine] = []
    for frags in rows:
        text_line = "".join(part for _, part in frags)
        has_style = any(style for style, _ in frags)
        lines.append(
            TranscriptLine(
                "class:transcript.assistant",
                text_line,
                fragments=list(frags) if has_style else None,
            )
        )
    # drop trailing blank rows Rich emits from its final newline, but keep any
    # internal blank spacing lines between blocks
    while lines and not lines[-1].text:
        lines.pop()
    return lines or _plain_lines(stripped)
