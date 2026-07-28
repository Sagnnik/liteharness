"""Render facade for the interactive CLI.

The full-screen prompt_toolkit TUI is the only supported interactive surface.
This module is a thin facade: command/session code reports transcript events
through the public ``render_*`` functions, which route to the active sink
registered via :func:`set_sink`. A no-op sink is used while no TUI is attached
(during startup/shutdown and in headless tests) so callers never need to guard
against a missing sink.
"""

from __future__ import annotations

from contextlib import AbstractContextManager, nullcontext
from typing import Any, Iterable, Protocol

# ``AssistantStream`` lives in liteharness_cli.tui.stream next to ``TuiAssistantStream`` (the
# live assistant-stream class it coordinates reasoning with). Re-exported here so
# existing ``render.AssistantStream`` references keep resolving. Imported below
# the stdlib imports and above the local definitions to keep all imports
# grouped; safe because ``liteharness_cli.tui.stream`` only accesses ``liteharness_cli.tui.render`` lazily
# (via ``stream._active_sink``), so module-level load order has no cycle.
from liteharness_cli.tui.stream import AssistantStream


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
    def reserve_reasoning_slot(self, before_stream: Any | None = None) -> dict: ...
    def finalize_reasoning_slot(self, span: dict, text: str, *, elapsed: float) -> None: ...
    def append_reasoning(self, text: str, *, elapsed: float) -> None: ...
    def toggle_reasoning(self) -> None: ...
    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None: ...
    def append_tool_result(self, name: str, content: str, *, exit_status: str | None = None) -> None: ...
    def append_todos(self, todos: list[dict]) -> None: ...
    def append_diff(self, diff_text: str, *, title: str = "diff") -> None: ...
    def append_shell_output(self, content: str) -> None: ...
    def append_usage(self, usage: dict[str, Any]) -> None: ...
    def append_notice(self, title: str, *lines: str) -> None: ...
    def append_warning(self, text: str) -> None: ...
    def append_error(self, text: str) -> None: ...
    def append_panel(self, title: str, *lines: str) -> None: ...
    def append_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None: ...
    def thinking(self, label: str = "thinking") -> AbstractContextManager[Any]: ...
    def begin_turn(self) -> None: ...
    def finish_turn(self) -> None: ...
    def clear_transcript(self) -> None: ...
    async def ask_approval(self, name: str, args: dict) -> str: ...
    async def ask_questions(self, questions: list[dict]) -> list[dict]: ...
    async def ask_line(self, message: str) -> str: ...
    async def run_config(self) -> Any: ...


# --- sink registry ---------------------------------------------------------
_ACTIVE_SINK: RenderSink | None = None


def set_sink(sink: RenderSink | None) -> None:
    global _ACTIVE_SINK
    _ACTIVE_SINK = sink


def get_sink() -> RenderSink | None:
    return _ACTIVE_SINK


class _NullStream:
    """No-op ``StreamSink`` used while no TUI is attached."""

    def feed(self, chunk: str) -> None: ...

    def stop(self) -> None: ...


class _NullSink:
    """No-op sink used while no TUI is attached.

    Every method is a silent no-op (or returns an empty default) so callers
    can run headlessly without guarding each ``render_*`` call. The
    interactive TUI replaces this sink via :func:`set_sink` at startup.
    """

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

    def start_assistant_stream(self) -> StreamSink:
        return _NullStream()

    def append_assistant(self, text: str) -> None: ...

    def reserve_reasoning_slot(self, before_stream: Any | None = None) -> dict:
        return {"start": -1, "count": 0, "text": "", "elapsed": 0.0}

    def finalize_reasoning_slot(self, span: dict, text: str, *, elapsed: float) -> None: ...

    def append_reasoning(self, text: str, *, elapsed: float) -> None: ...

    def toggle_reasoning(self) -> None: ...

    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None: ...

    def append_tool_result(self, name: str, content: str, *, exit_status: str | None = None) -> None: ...

    def append_todos(self, todos: list[dict]) -> None: ...

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None: ...

    def append_shell_output(self, content: str) -> None: ...

    def append_usage(self, usage: dict[str, Any]) -> None: ...

    def append_notice(self, title: str, *lines: str) -> None: ...

    def append_warning(self, text: str) -> None: ...

    def append_error(self, text: str) -> None: ...

    def append_panel(self, title: str, *lines: str) -> None: ...

    def append_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None: ...

    def thinking(self, label: str = "thinking") -> AbstractContextManager[Any]:
        return nullcontext()

    def begin_turn(self) -> None: ...

    def finish_turn(self) -> None: ...

    def clear_transcript(self) -> None: ...

    async def ask_approval(self, name: str, args: dict) -> str:
        return "no"

    async def ask_questions(self, questions: list[dict]) -> list[dict]:
        return []

    async def ask_line(self, message: str) -> str:
        return ""

    async def run_config(self) -> Any:
        from liteharness_cli.tui.config_flow import ConfigResult

        result = ConfigResult()
        result.note("/config is available only in the interactive TUI.")
        return result


_NULL_SINK = _NullSink()


def _sink() -> RenderSink:
    return _ACTIVE_SINK or _NULL_SINK


# --- public facade ---------------------------------------------------------
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


def render_assistant_panel(text: str) -> None:
    if text.strip():
        _sink().append_assistant(text)


def render_reasoning(text: str, *, elapsed: float) -> None:
    _sink().append_reasoning(text, elapsed=elapsed)


def render_tool_calls(calls: list[dict[str, Any]]) -> None:
    _sink().append_tool_calls(calls)


def render_tool_result(name: str, content: str, *, exit_status: str | None = None) -> None:
    _sink().append_tool_result(name, content, exit_status=exit_status)


def render_todos(todos: Iterable[dict]) -> None:
    active = [todo for todo in (todos or []) if todo.get("status") != "completed"]
    _sink().append_todos(active)


def render_diff(diff_text: str, *, title: str = "diff") -> None:
    _sink().append_diff(diff_text, title=title)


def render_shell_output(content: str) -> None:
    _sink().append_shell_output(content)


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
    _sink().begin_turn()


def finish_turn() -> None:
    _sink().finish_turn()


def clear_transcript() -> None:
    _sink().clear_transcript()


async def ask_approval(name: str, args: dict) -> str:
    return await _sink().ask_approval(name, args)


async def ask_questions(questions: list[dict]) -> list[dict]:
    return await _sink().ask_questions(questions)


async def ask_line(message: str) -> str:
    return await _sink().ask_line(message)