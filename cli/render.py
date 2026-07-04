"""Render facade for the interactive CLI.

The full-screen TUI is the supported interactive surface. This module keeps a
small sink API so command/session code can report transcript events without
knowing whether it is writing to the active TUI or the terminal fallback used
during startup/shutdown.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from typing import Any, Iterable, Protocol

from rich.align import Align
from rich.box import ROUNDED, SIMPLE
from rich.markdown import Markdown
from rich.padding import Padding
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from cli.theme import USER_BOX_BG, console

_TOOL_RESULT_PREVIEW = 320


class StreamSink(Protocol):
    def feed(self, chunk: str) -> None: ...
    def stop(self) -> None: ...


class RenderSink(Protocol):
    def append_header(
        self,
        *,
        mode: str,
        model: str,
        approval: bool,
        autosave: bool,
        session_end_reflection: bool,
    ) -> None: ...

    def append_user(self, text: str) -> None: ...
    def start_assistant_stream(self) -> StreamSink: ...
    def append_assistant(self, text: str) -> None: ...
    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None: ...
    def append_tool_result(self, name: str, content: str, *, exit_status: str | None = None) -> None: ...
    def append_todos(self, todos: list[dict]) -> None: ...
    def append_diff(self, diff_text: str, *, title: str = "diff") -> None: ...
    def append_usage(self, usage: dict[str, Any]) -> None: ...
    def append_notice(self, title: str, *lines: str) -> None: ...
    def append_warning(self, text: str) -> None: ...
    def append_error(self, text: str) -> None: ...
    def append_panel(self, title: str, *lines: str) -> None: ...
    def append_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None: ...
    def thinking(self, label: str = "thinking") -> AbstractContextManager[Any]: ...
    def begin_turn(self) -> None: ...
    def finish_turn(self) -> None: ...
    async def ask_approval(self, name: str, args: dict) -> str: ...
    async def ask_questions(self, questions: list[dict]) -> list[dict]: ...
    async def ask_line(self, message: str) -> str: ...
    async def run_config(self) -> Any: ...


_ACTIVE_SINK: RenderSink | None = None


def set_sink(sink: RenderSink | None) -> None:
    global _ACTIVE_SINK
    _ACTIVE_SINK = sink


def get_sink() -> RenderSink | None:
    return _ACTIVE_SINK


def _sink() -> RenderSink:
    return _ACTIVE_SINK or _TERMINAL_SINK


def render_header(
    *,
    mode: str,
    model: str,
    approval: bool,
    autosave: bool,
    session_end_reflection: bool,
) -> None:
    _sink().append_header(
        mode=mode,
        model=model,
        approval=approval,
        autosave=autosave,
        session_end_reflection=session_end_reflection,
    )


def render_user_echo(text: str) -> None:
    if text.strip():
        _sink().append_user(text.strip())


class AssistantStream:
    def __init__(self) -> None:
        self._stream = _sink().start_assistant_stream()
        self._buffer: list[str] = []

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer.append(chunk)
        self._stream.feed(chunk)

    @property
    def text(self) -> str:
        return "".join(self._buffer)

    def stop(self) -> None:
        self._stream.stop()


def render_assistant_panel(text: str) -> None:
    if text.strip():
        _sink().append_assistant(text)


def render_tool_call(name: str, args: Any) -> None:
    render_tool_calls([{"name": name, "args": args if isinstance(args, dict) else {}}])


def render_tool_calls(calls: list[dict[str, Any]]) -> None:
    _sink().append_tool_calls(calls)


def render_tool_result(name: str, content: str, *, exit_status: str | None = None) -> None:
    _sink().append_tool_result(name, content, exit_status=exit_status)


def render_todos(todos: Iterable[dict]) -> None:
    active = [todo for todo in (todos or []) if todo.get("status") != "completed"]
    _sink().append_todos(active)


def diff_renderable(diff_text: str) -> Text:
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
    _sink().append_diff(diff_text, title=title)


def render_usage_footer(usage: dict[str, Any]) -> None:
    if usage:
        _sink().append_usage(usage)


def render_notice(message: str, *, title: str | None = None) -> None:
    _sink().append_notice(title or "notice", message)


def render_warning(message: str) -> None:
    _sink().append_warning(message)


def render_error(message: str) -> None:
    _sink().append_error(message)


def render_panel_text(message: str, *, title: str, style: str = "default") -> None:
    del style
    _sink().append_panel(title, *str(message).splitlines())


def render_table(
    *,
    title: str,
    columns: list[str],
    rows: list[list[str]],
) -> None:
    _sink().append_table(title, columns, rows)


def thinking(label: str = "thinking") -> AbstractContextManager[Any]:
    return _sink().thinking(label)


def begin_turn() -> None:
    if _ACTIVE_SINK is not None:
        _ACTIVE_SINK.begin_turn()


def finish_turn() -> None:
    if _ACTIVE_SINK is not None:
        _ACTIVE_SINK.finish_turn()


async def ask_approval(name: str, args: dict) -> str:
    if _ACTIVE_SINK is None:
        render_warning(f"approval unavailable without TUI; denied: {name}")
        return "no"
    return await _ACTIVE_SINK.ask_approval(name, args)


async def ask_questions(questions: list[dict]) -> list[dict]:
    if _ACTIVE_SINK is None:
        return []
    return await _ACTIVE_SINK.ask_questions(questions)


async def ask_line(message: str) -> str:
    if _ACTIVE_SINK is None:
        render_warning(f"prompt unavailable without TUI: {message}")
        return ""
    return await _ACTIVE_SINK.ask_line(message)


class _TerminalAssistantStream:
    def __init__(self, sink: "_TerminalSink") -> None:
        self._sink = sink
        self._buffer: list[str] = []
        self._status: Any = None

    def _status_text(self) -> Text:
        tail = ""
        for candidate in reversed("".join(self._buffer).splitlines()):
            if candidate.strip():
                tail = candidate.strip()
                break
        label = Text("writing", style="accent")
        if tail:
            width = max(20, console.size.width - 16)
            if len(tail) > width:
                tail = "..." + tail[-(width - 3):]
            label.append("  ", style="muted")
            label.append(tail, style="muted")
        return label

    def feed(self, chunk: str) -> None:
        self._buffer.append(chunk)
        if self._status is None:
            self._status = console.status(self._status_text(), spinner="dots", spinner_style="accent")
            self._status.start()
        else:
            self._status.update(self._status_text())

    def stop(self) -> None:
        if self._status is not None:
            self._status.stop()
            self._status = None
        text = "".join(self._buffer)
        if text.strip():
            self._sink.append_assistant(text)


class _TerminalSink:
    def append_header(
        self,
        *,
        mode: str,
        model: str,
        approval: bool,
        autosave: bool,
        session_end_reflection: bool,
    ) -> None:
        body = Table.grid(padding=(0, 2))
        body.add_column(justify="left")
        meta = Text()
        meta.append("model ", style="usage")
        meta.append(model, style="usage.value")
        meta.append("   approval ", style="usage")
        meta.append("on" if approval else "off", style="usage.value")
        meta.append("   autosave ", style="usage")
        meta.append("on" if autosave else "off", style="usage.value")
        meta.append("   session end reflection ", style="usage")
        meta.append("on" if session_end_reflection else "off", style="usage.value")
        meta.append("   mode ", style="usage")
        meta.append(mode, style="usage.value")
        body.add_row(Text("LiteHarness", style="header"))
        body.add_row(meta)
        console.print(Panel(body, box=ROUNDED, border_style="header.frame", padding=(0, 1)))

    def append_user(self, text: str) -> None:
        console.print(
            Padding(
                Align.left(Text(text, style=f"user on {USER_BOX_BG}")),
                (1, 2),
                style=f"on {USER_BOX_BG}",
                expand=True,
            )
        )

    def start_assistant_stream(self) -> StreamSink:
        return _TerminalAssistantStream(self)

    def append_assistant(self, text: str) -> None:
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

    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        from cli.tool_display import BATCHABLE_TOOL_CALLS, format_batched_tool_args, format_tool_args

        index = 0
        while index < len(calls):
            call = calls[index]
            name = str(call.get("name") or "?")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}

            if name in BATCHABLE_TOOL_CALLS:
                batch = [calls[index]]
                next_index = index + 1
                while next_index < len(calls) and str(calls[next_index].get("name") or "?") == name:
                    batch.append(calls[next_index])
                    next_index += 1
                self._print_tool_call(name, format_batched_tool_args(name, batch))
                index = next_index
                continue

            self._print_tool_call(name, format_tool_args(name, args))
            index += 1

    @staticmethod
    def _print_tool_call(name: str, args_text: str) -> None:
        line = Text()
        line.append("-> ", style="tool")
        line.append(name, style="tool")
        if args_text:
            line.append("   ", style="tool")
            line.append(args_text, style="tool.args")
        console.print(line)

    def append_tool_result(self, name: str, content: str, *, exit_status: str | None = None) -> None:
        from cli.tool_display import format_tool_result_preview, should_show_tool_result

        if not should_show_tool_result(name, content, exit_status=exit_status):
            return
        preview = format_tool_result_preview(name, content, limit=_TOOL_RESULT_PREVIEW)
        line = Text()
        line.append("  \\_ ", style="muted")
        if exit_status and exit_status != "ok":
            style = "error" if exit_status in {"error", "denied"} else "warning"
            line.append(f"[{exit_status}] ", style=style)
        line.append(preview, style="tool.result")
        console.print(line)

    def append_todos(self, todos: list[dict]) -> None:
        table = Table(box=SIMPLE, show_header=False, padding=(0, 1), expand=False)
        table.add_column(width=2)
        table.add_column()
        glyphs = {
            "completed": ("*", "diff.add"),
            "in_progress": ("~", "accent"),
            "pending": ("o", "muted"),
            "cancelled": ("x", "diff.del"),
        }
        for todo in todos:
            glyph, style = glyphs.get(str(todo.get("status", "")), ("o", "muted"))
            table.add_row(Text(glyph, style=style), Text(str(todo.get("content", ""))))
        console.print()
        console.print(Panel(table, title="todos", title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None:
        console.print(Panel(diff_renderable(diff_text), title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))

    def append_usage(self, usage: dict[str, Any]) -> None:
        line = Text()

        def add(label: str, value: str) -> None:
            if line.plain:
                line.append("   ", style="usage")
            line.append(label + " ", style="usage")
            line.append(value, style="usage.value")

        add("in", f"{int(usage.get('input_tokens', 0) or 0):,}")
        add("out", f"{int(usage.get('output_tokens', 0) or 0):,}")
        cached = int(usage.get("cached_input_tokens", 0) or 0)
        if cached:
            add("cached", f"{cached:,}")
        cost = usage.get("cost_usd")
        if cost:
            add("$", f"{float(cost):.4f}")
        console.print(line)

    def append_notice(self, title: str, *lines: str) -> None:
        message = "\n".join(lines)
        if title and title != "notice":
            console.print(Panel(Text(message, style="notice"), title=title, title_align="left", box=ROUNDED, border_style="notice.frame", padding=(0, 1)))
        else:
            console.print(Text(message, style="notice"))

    def append_warning(self, text: str) -> None:
        console.print(Text(text, style="warning"))

    def append_error(self, text: str) -> None:
        console.print(Text("error: " + text, style="error"))

    def append_panel(self, title: str, *lines: str) -> None:
        console.print(Panel(Text("\n".join(lines), style="usage.value"), title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))

    def append_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        table = Table(box=SIMPLE, show_header=True, header_style="table.header", expand=False)
        for header in headers:
            table.add_column(header)
        for row in rows:
            table.add_row(*row)
        console.print(Panel(table, title=title, title_align="left", box=ROUNDED, border_style="panel.frame", padding=(0, 1)))

    def thinking(self, label: str = "thinking") -> AbstractContextManager[Any]:
        return console.status(Text(label, style="accent"), spinner="dots", spinner_style="accent")

    def begin_turn(self) -> None:
        return

    def finish_turn(self) -> None:
        return

    async def ask_approval(self, name: str, args: dict) -> str:
        del args
        self.append_warning(f"approval unavailable without TUI; denied: {name}")
        return "no"

    async def ask_questions(self, questions: list[dict]) -> list[dict]:
        del questions
        return []

    async def ask_line(self, message: str) -> str:
        self.append_warning(f"prompt unavailable without TUI: {message}")
        return ""

    async def run_config(self) -> Any:
        from cli.config_panel import ConfigResult

        result = ConfigResult()
        result.note("/config is available only in the interactive TUI.")
        return result


_TERMINAL_SINK = _TerminalSink()
