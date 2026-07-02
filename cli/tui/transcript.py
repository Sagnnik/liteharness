from __future__ import annotations

from typing import Any

from cli.tool_display import (
    BATCHABLE_TOOL_CALLS,
    format_batched_tool_args,
    format_tool_args,
    should_show_tool_call,
    should_show_tool_result,
    spawn_subagent_result_summary,
)
from cli.tui.formatting import USER_STYLE, user_message_lines
from cli.tui.markdown_render import todos_transcript_lines
from cli.tui.models import TranscriptLine
from cli.tui.stream import Thinking, TuiAssistantStream
from cli.tui.utils import term_height, term_width

_TAG_STYLES: dict[str, str] = {
    "session": "class:transcript.tag.session",
    "mcp": "class:transcript.tag.mcp",
    "notice": "class:transcript.tag.notice",
    "skill": "class:transcript.tag.skill",
    "init": "class:transcript.tag.init",
    "save": "class:transcript.tag.save",
    "resume": "class:transcript.tag.session",
    "compaction": "class:transcript.tag.notice",
    "pre-execution checkpoint": "class:transcript.tag.notice",
    "warning": "class:transcript.tag.notice",
}


def _tag_style(title: str) -> str:
    key = title.lower()
    if key.startswith("question"):
        return "class:transcript.tag.session"
    return _TAG_STYLES.get(key, "class:transcript.tag.notice")


def _tagged_lines(title: str, *lines: str) -> list[TranscriptLine]:
    parts: list[str] = []
    for line in lines:
        parts.extend(str(line).splitlines() or [""])
    if not parts:
        parts = [""]
    indent = " " * (len(f"[{title}]") + 4)
    out: list[TranscriptLine] = []
    for index, body in enumerate(parts):
        if index == 0:
            text = f"[{title}]    {body}"
            out.append(
                TranscriptLine(
                    style="",
                    text=text,
                    fragments=[
                        (_tag_style(title), f"[{title}]"),
                        ("class:transcript.tag.body", f"    {body}"),
                    ],
                )
            )
        else:
            out.append(TranscriptLine(style="class:transcript.tag.body", text=indent + body))
    return out


class TranscriptMixin:
    """Transcript buffer, render-sink methods, and scroll behavior."""

    def append_header(
        self,
        *,
        mode: str,
        model: str,
        approval: bool,
        autosave: bool,
        session_end_reflection: bool,
    ) -> None:
        self.append_notice(
            "session",
            f"model {model}  approval {'on' if approval else 'off'}  autosave {'on' if autosave else 'off'}  session end reflection {'on' if session_end_reflection else 'off'}",
        )

    def append_user(self, text: str) -> None:
        if not text.strip():
            return
        width = self._transcript_render_width or term_width()
        self._append_transcript(*user_message_lines(text, width=width))
        self._layout_term_width = width

    def _on_transcript_render_width(self, width: int) -> None:
        self._on_transcript_render_size(width, self._transcript_viewport_height or self._transcript_viewport_lines())

    def _on_transcript_render_size(self, width: int, height: int) -> None:
        self._transcript_render_width = width
        self._transcript_viewport_height = height
        if self._transcript_store.set_width(width) and self._follow_transcript:
            self._scroll_transcript_to_bottom()

    def _after_render(self) -> None:
        width = self._transcript_render_width
        if width <= 0:
            return
        if not self._transcript_store.has_user_blocks:
            self._layout_term_width = width
            return
        if width == self._layout_term_width and self._user_blocks_fit_width(width):
            return
        self._reflow_user_blocks_for_width(width)

    def _expected_user_band_width(self, width: int | None = None) -> int:
        from cli.tui.formatting import user_band_width

        return user_band_width(width=width if width is not None else term_width())

    def _user_blocks_fit_width(self, width: int) -> bool:
        expected = self._expected_user_band_width(width)
        index = 0
        while index < len(self._lines):
            line = self._lines[index]
            if line.user_source is None:
                index += 1
                continue
            end = index + 1
            while end < len(self._lines) and self._lines[end].style == USER_STYLE:
                end += 1
            for row in self._lines[index:end]:
                if len(row.text) != expected:
                    return False
            index = end
        return True

    def _reflow_user_blocks_for_width(self, width: int) -> None:
        if width == self._layout_term_width and self._user_blocks_fit_width(width):
            return

        follow = self._follow_transcript
        old_scroll = self._transcript_pane.vertical_scroll if self._transcript_pane else 0

        new_lines: list[TranscriptLine] = []
        index = 0
        while index < len(self._lines):
            line = self._lines[index]
            if line.user_source is None:
                new_lines.append(line)
                index += 1
                continue

            end = index + 1
            while end < len(self._lines) and self._lines[end].style == USER_STYLE:
                end += 1

            block = user_message_lines(line.user_source, width=width)
            new_lines.extend(block)
            index = end

        self._transcript_store.reset(new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._layout_term_width = width

        self.invalidate()
        if follow:
            self._scroll_transcript_to_bottom()
        elif self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = min(old_scroll, self._max_transcript_scroll())

    def append_muted(self, text: str) -> None:
        self._append_transcript(TranscriptLine("class:transcript.muted", text))

    def append_notice(self, title: str, *lines: str) -> None:
        self._append_transcript(*_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", ""))

    def append_warning(self, text: str) -> None:
        self._append_transcript(*_tagged_lines("warning", str(text)), TranscriptLine("class:transcript.muted", ""))

    def append_error(self, text: str) -> None:
        self._append_transcript(
            TranscriptLine(
                style="",
                text=f"[error]    {text}",
                fragments=[
                    ("class:transcript.error", "[error]"),
                    ("class:transcript.tag.body", f"    {text}"),
                ],
            ),
            TranscriptLine("class:transcript.muted", ""),
        )

    def append_panel(self, title: str, *lines: str) -> None:
        self._append_transcript(*_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", ""))

    def append_table(self, title: str, headers: list[str], rows: list[list[str]]) -> None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines_out = [
            TranscriptLine("class:transcript.notice", title),
            TranscriptLine("class:transcript.panel", f"  {header_line}"),
            TranscriptLine("class:transcript.muted", "  " + "  ".join("-" * w for w in col_widths)),
        ]
        for row in rows:
            line = "  ".join(str(row[i]).ljust(col_widths[i]) for i in range(len(headers)))
            lines_out.append(TranscriptLine("class:transcript.panel", f"  {line}"))
        lines_out.append(TranscriptLine("class:transcript.muted", ""))
        self._append_transcript(*lines_out)

    def append_assistant(self, text: str) -> None:
        if not text.strip():
            return
        lines = [TranscriptLine("class:transcript.assistant", line) for line in text.strip().splitlines()]
        self._append_transcript(*lines, TranscriptLine("class:transcript.muted", ""))

    @staticmethod
    def _assistant_stream_lines(text: str) -> list[TranscriptLine]:
        if not text:
            return [TranscriptLine("class:transcript.assistant", "")]
        return [TranscriptLine("class:transcript.assistant", part) for part in text.split("\n")]

    def set_assistant_stream(
        self,
        text: str,
        start: int | None,
        count: int,
    ) -> tuple[int, int]:
        lines = self._assistant_stream_lines(text)
        if start is None:
            start = len(self._lines)
            self._transcript_store.append(lines)
        else:
            self._transcript_store.replace(start, count, lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()
        return start, len(lines)

    def finalize_assistant_stream(self, text: str, start: int | None, count: int) -> None:
        if start is None:
            self.append_assistant(text)
            return
        stripped = text.strip()
        if not stripped:
            self.clear_assistant_stream(start, count)
            return
        final_lines = [TranscriptLine("class:transcript.assistant", line) for line in stripped.splitlines()]
        self._transcript_store.replace(start, count, [*final_lines, TranscriptLine("class:transcript.muted", "")])
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def clear_assistant_stream(self, start: int | None, count: int) -> None:
        if start is None or count <= 0:
            return
        self._transcript_store.delete(start, count)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def _sync_transcript_buffer(self, *, scroll: bool = True, invalidate_ui: bool = True) -> None:
        self._transcript_store.reset()
        self._transcript_revision = self._transcript_store.revision
        if scroll:
            self._scroll_transcript_to_bottom()
        if invalidate_ui:
            self.invalidate()

    def append_tool_call(self, name: str, args: Any) -> None:
        self.append_tool_calls([{"name": name, "args": args if isinstance(args, dict) else {}}])

    def _append_tool_spacer(self) -> None:
        self._append_transcript(TranscriptLine("class:transcript.muted", ""))

    def _tool_spacer_line(self) -> TranscriptLine:
        return TranscriptLine("class:transcript.muted", "")

    def _advance_tool_batch(self, calls: list[dict[str, Any]], index: int, name: str) -> int:
        if name not in BATCHABLE_TOOL_CALLS:
            return index + 1
        next_index = index + 1
        while next_index < len(calls) and str(calls[next_index].get("name") or "?") == name:
            next_index += 1
        return next_index

    def append_tool_calls(self, calls: list[dict[str, Any]]) -> None:
        if not calls:
            return
        lines_out: list[TranscriptLine] = []
        index = 0
        while index < len(calls):
            call = calls[index]
            name = str(call.get("name") or "?")
            args = call.get("args") if isinstance(call.get("args"), dict) else {}

            if not should_show_tool_call(name):
                index = self._advance_tool_batch(calls, index, name)
                continue

            if name in BATCHABLE_TOOL_CALLS:
                batch = [calls[index]]
                next_index = index + 1
                while next_index < len(calls) and str(calls[next_index].get("name") or "?") == name:
                    batch.append(calls[next_index])
                    next_index += 1
                args_text = format_batched_tool_args(name, batch)
                lines_out.extend([self._tool_call_line(name, args_text), self._tool_spacer_line()])
                index = next_index
                continue

            if name == "todo":
                index += 1
                continue

            args_text = format_tool_args(name, args)
            parts = args_text.splitlines() or [""]
            for part in parts:
                lines_out.append(self._tool_call_line(name, part))
            lines_out.append(self._tool_spacer_line())
            index += 1
        self._append_transcript(*lines_out)

    def _tool_call_line(self, name: str, args_text: str) -> TranscriptLine:
        prefix = "→ "
        sep = "   " if args_text else ""
        text = f"{prefix}{name}{sep}{args_text}".rstrip()
        fragments = [("class:transcript.tool", f"{prefix}{name}")]
        if args_text:
            fragments.append(("class:transcript.tool.args", f"{sep}{args_text}"))
        return TranscriptLine("", text, fragments=fragments)

    def append_tool_result(self, name: str, content: str, *, exit_status: str | None = None) -> None:
        if not should_show_tool_result(name):
            return

        if name == "spawn_subagent":
            summary = spawn_subagent_result_summary(content)
            summary_text = f"  └ {summary}"
            self._append_transcript(
                TranscriptLine(
                    "class:transcript.subagent.summary",
                    summary_text,
                    fragments=[("class:transcript.subagent.summary", summary_text)],
                )
            )
            return

        preview = " ".join(str(content).split())
        if len(preview) > 320:
            preview = preview[:320] + "..."
        prefix = f"  [{exit_status}] " if exit_status and exit_status != "ok" else "  "
        body = prefix + preview
        self._append_transcript(
            TranscriptLine(
                "class:transcript.tool.result",
                body,
                fragments=[("class:transcript.tool.result", body)],
            )
        )

    def append_usage(self, usage: dict[str, Any]) -> None:
        self.invalidate()

    def append_todos(self, todos: list[dict]) -> None:
        if not todos:
            return
        width = self._transcript_render_width or term_width()
        self._append_transcript(*todos_transcript_lines(todos, width=width), TranscriptLine("class:transcript.muted", ""))

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None:
        self.append_panel(title, *str(diff_text).splitlines())

    def start_assistant_stream(self) -> TuiAssistantStream:
        return TuiAssistantStream(self)

    def thinking(self, label: str = "thinking") -> Thinking:
        return Thinking(self, label)

    def _transcript_plain_text(self) -> str:
        return self._transcript_store.plain_text()

    def _append_transcript(self, *lines: TranscriptLine) -> None:
        if not lines:
            return
        self._transcript_store.append(list(lines))
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def _transcript_content_lines(self) -> int:
        return self._transcript_store.total_rows

    def _chrome_height_lines(self) -> int:
        lines = 5
        if self._working_status_visible():
            lines += 1
        if self._form_visible():
            lines += 2
        if self._menu_header_fragments():
            lines += 1
        lines += self._menu_body_height()
        return lines

    def _transcript_viewport_lines(self) -> int:
        if self._transcript_viewport_height > 0:
            return self._transcript_viewport_height
        return max(1, term_height() - self._chrome_height_lines())

    def _max_transcript_scroll(self) -> int:
        return self._transcript_store.max_scroll(self._transcript_viewport_lines())

    def _set_transcript_scroll(self, value: int) -> None:
        if self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = max(0, min(value, self._max_transcript_scroll()))

    def _scroll_transcript_to_bottom(self) -> None:
        if self._follow_transcript and self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = self._max_transcript_scroll()

    def _scroll_transcript_by(self, delta: int) -> None:
        if self._transcript_pane is None:
            return
        self._transcript_pane.vertical_scroll = max(
            0, min(self._max_transcript_scroll(), self._transcript_pane.vertical_scroll + delta)
        )
        if delta < 0:
            self._follow_transcript = False
        elif self._transcript_pane.vertical_scroll >= self._max_transcript_scroll():
            self._follow_transcript = True

    def _scroll_transcript_to_top(self) -> None:
        if self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = 0
            self._follow_transcript = False

    def _resume_transcript_follow(self) -> None:
        self._follow_transcript = True
        self._scroll_transcript_to_bottom()
