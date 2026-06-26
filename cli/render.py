"""All Rich rendering for the LiteHarness CLI.

Every visible element routes through here so the palette stays consistent.
Sections: header, user echo, assistant (streamed panel), tool calls/results,
todos, diffs, per-turn usage footer, notices and errors.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from rich.box import HEAVY, ROUNDED, SIMPLE
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.theme import console

_TOOL_RESULT_PREVIEW = 320
_TOOL_ARGS_PREVIEW = 200


# --- header -----------------------------------------------------------------
def render_header(*, mode: str, model: str, approval: bool, autosave: bool) -> None:
    body = Table.grid(padding=(0, 2))
    body.add_column(justify="left")
    title = Text("LiteHarness", style="header")
    meta = Text()
    meta.append("model ", style="usage")
    meta.append(model, style="usage.value")
    meta.append("   mode ", style="usage")
    meta.append(mode, style="mode.plan" if mode == "plan" else "mode.normal")
    meta.append("   approval ", style="usage")
    meta.append("on" if approval else "off", style="usage.value")
    meta.append("   autosave ", style="usage")
    meta.append("on" if autosave else "off", style="usage.value")
    body.add_row(title)
    body.add_row(meta)
    console.print(Panel(body, box=ROUNDED, border_style="header.frame", padding=(0, 1)))
    console.print(Text("Shift+Tab toggles plan/normal  •  /menu for commands  •  /help", style="muted"))


# --- user echo --------------------------------------------------------------
def render_user_echo(text: str) -> None:
    """Echo the user prompt in a clearly highlighted gray section."""
    if not text.strip():
        return
    console.print()
    console.print(
        Panel(
            Text(text.strip(), style="user"),
            title="you",
            title_align="left",
            box=HEAVY,
            border_style="user.frame",
            padding=(0, 1),
        )
    )


# --- assistant streaming ----------------------------------------------------
class AssistantStream:
    """Accumulates streamed assistant text, showing live progress on a single
    status line, then prints the finished Markdown panel once.

    A multi-line Rich ``Live`` region is deliberately avoided for the streamed
    body: once the text grows past the terminal height, ``Live`` repaints a full
    screen (cursor-up + erase-line per visible row) on every refresh, which
    flickers badly for long output such as plan mode. A single-line status
    spinner updates in place without that cost, and the formatted panel is
    rendered a single time when the message completes.
    """

    def __init__(self) -> None:
        self._buffer: list[str] = []
        self._status: Any = None
        self.started = False

    def _status_text(self) -> Text:
        # Collapse the streamed text to its last non-empty line so the status row
        # stays a single line (multi-line status would reintroduce flicker).
        tail = ""
        for candidate in reversed("".join(self._buffer).splitlines()):
            if candidate.strip():
                tail = candidate.strip()
                break
        label = Text("writing", style="accent")
        if tail:
            width = max(20, console.size.width - 16)
            if len(tail) > width:
                tail = "…" + tail[-(width - 1):]
            label.append("  ", style="muted")
            label.append(tail, style="muted")
        return label

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer.append(chunk)
        if not self.started:
            self._status = console.status(self._status_text(), spinner="dots", spinner_style="accent")
            self._status.start()
            self.started = True
        elif self._status is not None:
            self._status.update(self._status_text())

    @property
    def text(self) -> str:
        return "".join(self._buffer)

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        text = "".join(self._buffer)
        if text.strip():
            render_assistant_panel(text)


def render_assistant_panel(text: str) -> None:
    """Render a complete (non-streamed) assistant message."""
    if not text.strip():
        return
    console.print()
    console.print(
        Panel(
            Markdown(text),
            title="assistant",
            title_align="left",
            box=ROUNDED,
            border_style="assistant.frame",
            padding=(0, 1),
        )
    )


# --- tools ------------------------------------------------------------------
def render_tool_call(name: str, args: Any) -> None:
    arg_text = _format_args(args)
    line = Text()
    line.append("→ ", style="tool")
    line.append(name, style="tool")
    if arg_text:
        line.append("  ", style="tool")
        line.append(arg_text, style="tool.args")
    console.print(line)


def render_tool_result(name: str, content: str, *, exit_status: str | None = None) -> None:
    preview = " ".join(str(content).split())
    truncated = preview[:_TOOL_RESULT_PREVIEW]
    suffix = "…" if len(preview) > _TOOL_RESULT_PREVIEW else ""
    line = Text()
    line.append("  └ ", style="muted")
    if exit_status and exit_status not in {"ok"}:
        line.append(f"[{exit_status}] ", style="error" if exit_status in {"error", "denied"} else "warning")
    line.append(truncated + suffix, style="tool.result")
    console.print(line)


def _format_args(args: Any) -> str:
    if isinstance(args, dict):
        if not args:
            return ""
        text = json.dumps(args, ensure_ascii=False)
    else:
        text = str(args)
    text = " ".join(text.split())
    return text[:_TOOL_ARGS_PREVIEW] + ("…" if len(text) > _TOOL_ARGS_PREVIEW else "")


# --- todos ------------------------------------------------------------------
_STATUS_GLYPH = {
    "completed": ("●", "diff.add"),
    "in_progress": ("◐", "accent"),
    "pending": ("○", "muted"),
    "cancelled": ("✗", "diff.del"),
}


def render_todos(todos: Iterable[dict]) -> None:
    todos = list(todos or [])
    if not todos:
        return
    table = Table(box=SIMPLE, show_header=False, padding=(0, 1), expand=False)
    table.add_column(width=2)
    table.add_column()
    for todo in todos:
        glyph, style = _STATUS_GLYPH.get(str(todo.get("status", "")), ("○", "muted"))
        content = str(todo.get("content", ""))
        style_text = "muted" if todo.get("status") == "completed" else "default"
        table.add_row(Text(glyph, style=style), Text(content, style=style_text))
    console.print()
    console.print(Panel(table, title="todos", title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))


# --- diffs ------------------------------------------------------------------
def diff_renderable(diff_text: str) -> Text:
    """Color a unified diff (green additions, red deletions)."""
    out = Text()
    for line in str(diff_text).splitlines():
        if line.startswith(("+++", "---")):
            out.append(line + "\n", style="diff.meta")
        elif line.startswith("@@"):
            out.append(line + "\n", style="diff.hunk")
        elif line.startswith("+"):
            out.append(line + "\n", style="diff.add")
        elif line.startswith("-"):
            out.append(line + "\n", style="diff.del")
        else:
            out.append(line + "\n", style="tool.result")
    return out


def render_diff(diff_text: str, *, title: str = "diff") -> None:
    console.print(Panel(diff_renderable(diff_text), title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))


# --- usage footer -----------------------------------------------------------
def render_usage_footer(usage: dict[str, Any]) -> None:
    """One gray line summarizing the tokens/cost of the last turn."""
    if not usage:
        return
    line = Text()

    def add(label: str, value: str) -> None:
        if line.plain:
            line.append("   ", style="usage")
        line.append(label + " ", style="usage")
        line.append(value, style="usage.value")

    inp = int(usage.get("input_tokens", 0) or 0)
    out = int(usage.get("output_tokens", 0) or 0)
    cached = int(usage.get("cached_input_tokens", 0) or 0)
    cost = usage.get("cost_usd")
    add("↑ in", f"{inp:,}")
    add("↓ out", f"{out:,}")
    if cached:
        add("⟳ cached", f"{cached:,}")
    if cost:
        add("$", f"{float(cost):.4f}")
    console.print(line)


# --- notices / errors -------------------------------------------------------
def render_notice(message: str, *, title: str | None = None) -> None:
    if title:
        console.print(Panel(Text(message, style="notice"), title=title, title_align="left", box=ROUNDED, border_style="notice.frame", padding=(0, 1)))
    else:
        console.print(Text(message, style="notice"))


def render_warning(message: str) -> None:
    console.print(Text(message, style="warning"))


def render_error(message: str) -> None:
    console.print(Text("✗ " + message, style="error"))


def render_panel_text(message: str, *, title: str, style: str = "default") -> None:
    console.print(Panel(Text(message, style=style), title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))


def render_table(
    *,
    title: str,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    table = Table(box=SIMPLE, show_header=True, header_style="table.header", expand=False)
    for col in columns:
        table.add_column(col)
    for row in rows:
        table.add_row(*row)
    console.print(Panel(table, title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))


def thinking(label: str = "thinking"):
    """Cyan spinner context manager used while awaiting the model."""
    return console.status(Text(label, style="accent"), spinner="dots", spinner_style="accent")
