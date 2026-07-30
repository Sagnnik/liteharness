from __future__ import annotations

from typing import Any

from liteharness_cli.tui.tool_display import (
    BATCHABLE_TOOL_CALLS,
    format_batched_tool_args,
    format_tool_args,
    format_tool_result_preview,
    should_show_tool_call,
    should_show_tool_result,
)
from liteharness_cli.tui.formatting import USER_STYLE, user_message_lines
from liteharness_cli.tui.markdown import (
    _REASONING_COLLAPSED_STYLE,
    _diff_transcript_lines,
    _reasoning_block_lines,
    _shell_output_lines,
    _tagged_lines,
    markdown_transcript_lines,
    todos_transcript_lines,
)
from liteharness_cli.tui.models import TranscriptLine
from liteharness_cli.tui.stream import Thinking, TuiAssistantStream
from liteharness_cli.tui.utils import term_height, term_width


# Module-level TranscriptLine row builders (``_diff_transcript_lines``,
# ``_shell_output_lines``, ``_tagged_lines``, ``_reasoning_block_lines`` and
# their helpers) live in liteharness_cli.tui.markdown alongside ``markdown_transcript_lines``
# and ``todos_transcript_lines``, so every "build TranscriptLines from data"
# routine sits in one module. Imported back here for the TranscriptMixin sink
# methods below.


def _header_project() -> str:
    """CWD-with-branch string for the header's Project cell."""
    from liteharness_cli.tui.utils import display_cwd

    return display_cwd()


def _header_addons_summary(mcp, skill_loader) -> str:
    """Summarize active MCP servers + skills for the header's Add-ons cell.

    Reads the TuiApp-held MCPManager and the coding session's SkillLoader;
    both are optional so headless/test paths (no MCP, no skills dir) render
    an empty summary instead of failing.
    """
    parts: list[str] = []
    try:
        if mcp is not None:
            server_names = sorted(
                name
                for name, info in mcp.servers.items()
                if info.get("status") != "error"
            )
            n_mcp = len(server_names)
            if n_mcp:
                names = ", ".join(server_names[:3])
                if len(server_names) > 3:
                    names += ", …"
                parts.append(f"{n_mcp} MCPs ({names})" if names else f"{n_mcp} MCPs")
    except Exception:
        pass
    try:
        if skill_loader is not None:
            parts.append(f"{len(skill_loader.load())} Skills")
    except Exception:
        pass
    return ", ".join(parts)


def _header_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version

        try:
            return version("liteharness")
        except PackageNotFoundError:
            return "dev"
    except Exception:
        return "dev"
    except Exception:
        return "dev"


class TranscriptMixin:
    """Transcript buffer, render-sink methods, and scroll behavior.

    Composed into ``TuiApp`` (see cli/app.py) and registered as the active
    ``RenderSink`` via ``render.set_sink``. Methods are grouped by concern
    and separated by banner comments inline:

    - Tagged/diff/shell/reasoning line builders       (module scope: liteharness_cli.tui.markdown)
    - Sink entry points (high-level render calls)   (append_header, append_user)
    - Resize + reflow machinery                     (_on_transcript_render_width...)
    - Tagged render sink methods                     (append_notice, append_warning, ...)
    - Assistant markdown rendering                  (append_assistant, _reasoning_block_for_span)
    - Reasoning slot lifecycle                       (reserve_reasoning_slot, finalize, toggle)
    - Live assistant streaming                       (set_assistant_stream, finalize, clear)
    - Transcript buffer primitives + reset           (_append_transcript, _sync...)
    - Tool calls & results                          (append_tool_calls, append_tool_result)
    - Todos / diff / shell output                   (append_todos, append_diff, append_shell_output)
    - Streaming adapters (start_assistant_stream, thinking)
    - Layout sizing + scroll navigation             (_chrome_height_lines, _scroll_*)
    """

    # ------------------------------------------------------------------ #
    # Sink entry points (high-level render)                               #
    # ------------------------------------------------------------------ #
    def append_header(
        self,
        *,
        mode: str,
        model: str,
        approval: bool,
        yolo: bool = False,
        autosave: bool,
        session_end_reflection: bool,
    ) -> None:
        del (
            autosave,
            session_end_reflection,
        )  # surfaced elsewhere (not in the new header)
        width = self._transcript_render_width or term_width()
        from liteharness_cli.tui.header import header_lines

        source = {
            "mode": mode,
            "model": model,
            "approval": approval,
            "yolo": yolo,
            "project": _header_project(),
            # getattr chains: bare TranscriptMixin harnesses (tests) have no
            # mcp/coding wired; both summary inputs degrade to empty.
            "addons_summary": _header_addons_summary(
                getattr(self, "mcp", None),
                getattr(getattr(self, "coding", None), "skill_loader", None),
            ),
            "version": _header_version(),
        }
        rows = header_lines(width=width, show_logo=width >= 96, **source)
        # The trailing blank line is part of the tracked block so an in-place
        # replace removes the old blank too (otherwise spacers accumulate).
        if rows:
            block_lines = [*rows, TranscriptLine("class:transcript.muted", "")]
        else:
            # narrow-terminal fallback (<40 cols): a single [session] notice,
            # still tracked so a later resize regenerates the full header.
            block_lines = [
                *_tagged_lines(
                    "session", f"model {model}  approval {'on' if approval else 'off'}"
                ),
                TranscriptLine("class:transcript.muted", ""),
            ]

        if self._header_block is None:
            # First render: append at the top of an (initially empty) transcript.
            start = len(self._transcript_store.lines)
            self._transcript_store.append(block_lines)
            self._header_block = {
                "start": start,
                "count": len(block_lines),
                "width": width,
                "source": source,
            }
        else:
            # Subsequent renders (e.g. /config changed the model/mode): replace
            # the existing top-of-transcript block in place instead of appending
            # a duplicate banner mid-conversation.
            start = self._header_block["start"]
            old_count = self._header_block["count"]
            self._transcript_store.replace(start, old_count, block_lines)
            self._header_block["count"] = len(block_lines)
            self._header_block["width"] = width
            self._header_block["source"] = source
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_user(self, text: str) -> None:
        if not text.strip():
            return
        width = self._transcript_render_width or term_width()
        # one blank (non-bg) line separates the gray user band from the next render,
        # matching the muted spacer convention used by every other transcript block
        self._append_transcript(*user_message_lines(text, width=width))
        self._append_transcript(TranscriptLine("class:transcript.muted", ""))
        self._layout_term_width = width

    # ------------------------------------------------------------------ #
    # Resize + reflow machinery                                          #
    # ------------------------------------------------------------------ #
    def _on_transcript_render_width(self, width: int) -> None:
        self._on_transcript_render_size(
            width, self._transcript_viewport_height or self._transcript_viewport_lines()
        )

    def _on_transcript_render_size(self, width: int, height: int) -> None:
        self._transcript_render_width = width
        self._transcript_viewport_height = height
        if width > 0:
            self._transcript_ready.set()
        if self._transcript_store.set_width(width):
            # Re-flow the tracked header block at the new width so the
            # rounded dashboard / logo don't wrap into a half-screen
            # artifact on terminal shrink (and re-tighten on grow).
            self._reflow_header_for_width(width)
            if self._follow_transcript:
                self._scroll_transcript_to_bottom()

    def _reflow_header_for_width(self, width: int) -> None:
        """Regenerate the tracked header block at ``width`` if it changed.

        Mirrors ``_reflow_user_blocks_for_width`` but for the single tracked
        header block: rebuilds its ``TranscriptLine`` rows from the stored
        source kwargs at the new width and ``replace``s it in place. Guards
        keep this a no-op when no header has been rendered yet or the width
        is unchanged / below the render threshold.
        """
        block = self._header_block
        if block is None or width == block["width"]:
            return
        from liteharness_cli.tui.header import header_lines

        rows = header_lines(width=width, show_logo=width >= 96, **block["source"])
        if rows:
            new_lines = [*rows, TranscriptLine("class:transcript.muted", "")]
        else:
            new_lines = [
                *_tagged_lines(
                    "session",
                    f"model {block['source']['model']}  "
                    f"approval {'on' if block['source']['approval'] else 'off'}",
                ),
                TranscriptLine("class:transcript.muted", ""),
            ]
        self._transcript_store.replace(block["start"], block["count"], new_lines)
        block["count"] = len(new_lines)
        block["width"] = width
        self._transcript_revision = self._transcript_store.revision

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
        from liteharness_cli.tui.formatting import user_band_width

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
        old_scroll = (
            self._transcript_pane.vertical_scroll if self._transcript_pane else 0
        )

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
            self._transcript_pane.vertical_scroll = min(
                old_scroll, self._max_transcript_scroll()
            )

    # Tagged render sink methods (notice/warning/error/panel/table) ------------
    def append_notice(self, title: str, *lines: str) -> None:
        self._append_transcript(
            *_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", "")
        )

    def append_warning(self, text: str) -> None:
        self._append_transcript(
            *_tagged_lines("warning", str(text)),
            TranscriptLine("class:transcript.muted", ""),
        )

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
        self._append_transcript(
            *_tagged_lines(title, *lines), TranscriptLine("class:transcript.muted", "")
        )

    def append_table(
        self, title: str, headers: list[str], rows: list[list[str]]
    ) -> None:
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))
        header_line = "  ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines_out = [
            TranscriptLine("class:transcript.notice", title),
            TranscriptLine("class:transcript.panel", f"  {header_line}"),
            TranscriptLine(
                "class:transcript.muted", "  " + "  ".join("-" * w for w in col_widths)
            ),
        ]
        for row in rows:
            line = "  ".join(
                str(row[i]).ljust(col_widths[i]) for i in range(len(headers))
            )
            lines_out.append(TranscriptLine("class:transcript.panel", f"  {line}"))
        lines_out.append(TranscriptLine("class:transcript.muted", ""))
        self._append_transcript(*lines_out)

    # Assistant markdown rendering -----------------------------------------
    def append_assistant(self, text: str) -> None:
        if not text.strip():
            return
        width = self._transcript_render_width or term_width()
        lines = markdown_transcript_lines(text, width=width)
        self._append_transcript(*lines, TranscriptLine("class:transcript.muted", ""))

    def _reasoning_block_for_span(
        self, span: dict, *, expanded: bool | None = None
    ) -> list[TranscriptLine]:
        width = self._transcript_render_width or term_width()
        if expanded is None:
            expanded = self._show_reasoning
        return _reasoning_block_lines(
            span.get("text", ""),
            elapsed=float(span.get("elapsed", 0.0)),
            expanded=expanded,
            width=width,
        )

    # Reasoning slot lifecycle ---------------------------------------------
    def reserve_reasoning_slot(
        self, before_stream: TuiAssistantStream | None = None
    ) -> dict:
        """Insert a ``Thinking…`` placeholder above the live assistant stream.

        Called from the turn renderer on the first reasoning chunk of
        an LLM call. The placeholder sits before ``before_stream._line_start``
        (if the assistant stream already reserved its slot) or at the current
        end of the transcript otherwise; ``before_stream._line_start`` is
        shifted so the assistant finalize later targets the same lines.
        """
        anchor = len(self._transcript_store.lines)
        if before_stream is not None and before_stream._line_start is not None:
            anchor = before_stream._line_start
        placeholder = TranscriptLine(
            _REASONING_COLLAPSED_STYLE,
            " Thinking…",
            fragments=[(_REASONING_COLLAPSED_STYLE, " Thinking…")],
        )
        self._transcript_store.insert(anchor, [placeholder])
        if before_stream is not None and before_stream._line_start is not None:
            before_stream.shift_start(1)
        span = {"start": anchor, "count": 1, "text": "", "elapsed": 0.0}
        self._reasoning_spans.append(span)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()
        return span

    def finalize_reasoning_slot(self, span: dict, text: str, *, elapsed: float) -> None:
        span["text"] = text
        span["elapsed"] = float(elapsed)
        new_lines = self._reasoning_block_for_span(span)
        self._transcript_store.replace(span["start"], span["count"], new_lines)
        span["count"] = len(new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def toggle_reasoning(self) -> None:
        """Flip ``_show_reasoning`` and re-emit every reasoning span.

        Walks spans in ascending ``start`` order so the in-place ``replace``
        on an earlier span can shift every later span's lines. A running
        ``delta`` adjusts the later spans' ``start`` by the cumulative line
        count change of all already-processed earlier spans; ``replace``
        itself reassigns only the current span's ``self.lines`` slice, so the
        line objects of un-processed later spans keep their identities but
        move to higher indices when an earlier span grows. Without the delta
        accumulator those later spans would read stale ``start`` indices and
        either replace the wrong region or leave gaps.
        """
        self._show_reasoning = not self._show_reasoning
        spans = sorted(self._reasoning_spans, key=lambda s: s["start"])
        if not spans:
            self.invalidate()
            return
        delta = 0
        for span in spans:
            start = span["start"] + delta
            new_lines = self._reasoning_block_for_span(span)
            self._transcript_store.replace(start, span["count"], new_lines)
            delta += len(new_lines) - span["count"]
            span["start"] = start
            span["count"] = len(new_lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_reasoning(self, text: str, *, elapsed: float) -> None:
        """Append a finalized reasoning block at the end of the transcript.

        Used on the cancel-finalize path where there is no live assistant
        stream to interleave above; the block simply appends.
        """
        if not text.strip():
            return
        span = {
            "start": len(self._transcript_store.lines),
            "count": 0,
            "text": text,
            "elapsed": float(elapsed),
        }
        new_lines = self._reasoning_block_for_span(span)
        self._transcript_store.append(new_lines)
        span["start"] = len(self._transcript_store.lines) - len(new_lines)
        span["count"] = len(new_lines)
        self._reasoning_spans.append(span)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    @staticmethod
    def _assistant_stream_lines(text: str) -> list[TranscriptLine]:
        # Live-stream stays plain (cheap, smooth incremental paint); the finalized
        # markdown styling is swapped in by finalize_assistant_stream on completion.
        if not text:
            return [TranscriptLine("class:transcript.assistant", "")]
        return [
            TranscriptLine("class:transcript.assistant", part)
            for part in text.split("\n")
        ]

    # Live assistant streaming ----------------------------------------------
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

    def finalize_assistant_stream(
        self, text: str, start: int | None, count: int
    ) -> None:
        if start is None:
            self.append_assistant(text)
            return
        stripped = text.strip()
        if not stripped:
            self.clear_assistant_stream(start, count)
            return
        width = self._transcript_render_width or term_width()
        final_lines = markdown_transcript_lines(stripped, width=width)
        self._transcript_store.replace(
            start, count, [*final_lines, TranscriptLine("class:transcript.muted", "")]
        )
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

    # Transcript reset / clear ---------------------------------------------
    def _sync_transcript_buffer(
        self, *, scroll: bool = True, invalidate_ui: bool = True
    ) -> None:
        self._transcript_store.reset([])
        self._transcript_revision = self._transcript_store.revision
        if scroll:
            self._scroll_transcript_to_bottom()
        if invalidate_ui:
            self.invalidate()

    def clear_transcript(self) -> None:
        # Reset every index that points into the old transcript before the
        # store revision changes. Session switches, rollback, and /new all
        # share this path.
        self._header_block = None
        self._todos_block_start = None
        self._todos_block_count = 0
        self._reasoning_spans = []
        self._sync_transcript_buffer()

    # Tool calls & results -------------------------------------------------
    def _tool_spacer_line(self) -> TranscriptLine:
        return TranscriptLine("class:transcript.muted", "")

    def _advance_tool_batch(
        self, calls: list[dict[str, Any]], index: int, name: str
    ) -> int:
        if name not in BATCHABLE_TOOL_CALLS:
            return index + 1
        next_index = index + 1
        while (
            next_index < len(calls)
            and str(calls[next_index].get("name") or "?") == name
        ):
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
                while (
                    next_index < len(calls)
                    and str(calls[next_index].get("name") or "?") == name
                ):
                    batch.append(calls[next_index])
                    next_index += 1
                args_text = format_batched_tool_args(name, batch)
                lines_out.extend(
                    [self._tool_call_line(name, args_text), self._tool_spacer_line()]
                )
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

    def append_tool_result(
        self, name: str, content: str, *, exit_status: str | None = None
    ) -> None:
        if not should_show_tool_result(name, content, exit_status=exit_status):
            return

        preview = format_tool_result_preview(name, content)
        prefix = (
            f"  [{exit_status}] " if exit_status and exit_status != "ok" else "  └ "
        )
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

    # Todos / diff / shell output ------------------------------------------
    def append_todos(self, todos: list[dict]) -> None:
        if not todos:
            if self._todos_block_start is not None:
                self._transcript_store.delete(
                    self._todos_block_start, self._todos_block_count
                )
                self._todos_block_start = None
                self._todos_block_count = 0
                self._transcript_revision = self._transcript_store.revision
                self.invalidate()
            return

        width = self._transcript_render_width or term_width()
        lines = [
            *todos_transcript_lines(todos, width=width),
            TranscriptLine("class:transcript.muted", ""),
        ]
        if self._todos_block_start is not None:
            self._transcript_store.replace(
                self._todos_block_start, self._todos_block_count, lines
            )
        else:
            self._todos_block_start = len(self._transcript_store.lines)
            self._transcript_store.append(lines)
        self._todos_block_count = len(lines)
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    def append_diff(self, diff_text: str, *, title: str = "diff") -> None:
        self._append_transcript(*_diff_transcript_lines(diff_text, title))

    def append_shell_output(self, content: str) -> None:
        from liteharness_cli.tui.tool_display import format_shell_output

        header, body = format_shell_output(content)
        title = f"shell {header}".strip()
        body_lines = body.splitlines() if body.strip() else ["(no output)"]
        self._append_transcript(*_shell_output_lines(title, body_lines))

    def append_subagent_output(self, content: str) -> None:
        from liteharness_cli.tui.tool_display import format_subagent_output

        header, body = format_subagent_output(content)
        body_lines = body.splitlines() if body.strip() else ["(no output)"]
        lines = _shell_output_lines(header, body_lines)
        # Prefer the dedicated subagent summary style for body rows.
        styled = [lines[0]]
        for line in lines[1:-1]:
            styled.append(
                TranscriptLine(
                    "class:transcript.subagent.summary",
                    line.text,
                    fragments=[("class:transcript.subagent.summary", line.text)],
                )
            )
        if lines:
            styled.append(lines[-1])
        self._append_transcript(*styled)

    # Streaming adapters ---------------------------------------------------
    def start_assistant_stream(self) -> TuiAssistantStream:
        return TuiAssistantStream(self)

    def thinking(self, label: str = "thinking") -> Thinking:
        return Thinking(self, label)

    # Transcript buffer helpers (read/append primitives) -------------------
    def _append_transcript(self, *lines: TranscriptLine) -> None:
        if not lines:
            return
        self._transcript_store.append(list(lines))
        self._transcript_revision = self._transcript_store.revision
        self._scroll_transcript_to_bottom()
        self.invalidate()

    # ------------------------------------------------------------------ #
    # Layout sizing + scroll navigation                                   #
    # ------------------------------------------------------------------ #
    def _chrome_height_lines(self) -> int:
        lines = 4 + self._input_row_count()
        if self._working_status_visible():
            lines += 1
        if self._queue_line_visible():
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
            self._transcript_pane.vertical_scroll = max(
                0, min(value, self._max_transcript_scroll())
            )

    def _scroll_transcript_to_bottom(self) -> None:
        if self._follow_transcript and self._transcript_pane is not None:
            self._transcript_pane.vertical_scroll = self._max_transcript_scroll()

    def _scroll_transcript_by(self, delta: int) -> None:
        if self._transcript_pane is None:
            return
        self._transcript_pane.vertical_scroll = max(
            0,
            min(
                self._max_transcript_scroll(),
                self._transcript_pane.vertical_scroll + delta,
            ),
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
