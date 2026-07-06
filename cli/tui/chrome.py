from __future__ import annotations

import asyncio
import time

from prompt_toolkit.layout import HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import BufferControl, FormattedTextControl
from prompt_toolkit.layout.dimension import Dimension as D
from prompt_toolkit.layout.processors import AfterInput, BeforeInput

from cli.tui.constants import (
    FORM_FIELD_WIDTH,
    INPUT_MAX_ROWS_CAP,
    INPUT_MAX_ROWS_FRACTION,
    PICKER_MODES,
)
from cli.tui.formatting import worked_fragments, working_fragments
from cli.tui.utils import context_bar, display_cwd, model_footer_name, term_height, term_width
from cli.tui.widgets import TranscriptViewportControl
from config import cost_tracker
from model import active_model_name, active_reasoning_effort


class ChromeMixin:
    """Layout construction and bottom chrome fragment renderers."""

    def _build_layout(self) -> Layout:
        self._transcript_control = TranscriptViewportControl(
            self._transcript_store,
            get_scroll=lambda: self._transcript_pane.vertical_scroll if self._transcript_pane else 0,
            set_scroll=self._set_transcript_scroll,
            on_scroll=self._scroll_transcript_by,
            on_render_size=self._on_transcript_render_size,
            focus=self._focus_transcript,
            invalidate=self.invalidate,
        )
        self._transcript_inner = Window(
            content=self._transcript_control,
            height=D(weight=1),
            wrap_lines=False,
            style="class:screen",
            always_hide_cursor=True,
        )
        self._transcript_pane = self._transcript_inner
        self._input_window = Window(
            content=BufferControl(
                buffer=self._buffer,
                focusable=True,
                focus_on_click=True,
                input_processors=[BeforeInput(self._input_prefix_fragments)],
            ),
            height=self._input_height,
            wrap_lines=True,
            style="class:screen",
        )
        self._working_window = Window(
            content=FormattedTextControl(self._working_status_fragments),
            height=lambda: D.exact(1) if self._working_status_visible() else D.exact(0),
            style="class:screen",
        )
        self._queue_window = Window(
            content=FormattedTextControl(self._queue_fragments),
            height=lambda: D.exact(1) if self._queue_line_visible() else D.exact(0),
            style="class:screen",
        )
        self._form_pad_window = Window(
            content=FormattedTextControl(lambda: []),
            height=D.exact(1),
            width=lambda: D.exact(self._form_row_left_pad()) if self._form_visible() else D.exact(0),
            style="class:screen",
        )
        self._form_label_window = Window(
            content=FormattedTextControl(self._form_label_fragments),
            height=D.exact(1),
            width=lambda: D.exact(self._form_label_display_width()) if self._form_visible() else D.exact(0),
            style="class:screen",
        )
        self._form_field_window = Window(
            content=BufferControl(
                buffer=self._form_buffer,
                focusable=True,
                input_processors=[
                    BeforeInput(self._form_field_prefix_fragments),
                    AfterInput(self._form_field_suffix_fragments),
                ],
            ),
            height=D.exact(1),
            width=lambda: D.exact(self._form_field_outer_width()) if self._form_visible() else D.exact(0),
            style="class:screen",
        )
        self._form_row = VSplit(
            [self._form_pad_window, self._form_label_window, self._form_field_window],
            height=lambda: D.exact(1) if self._form_visible() else D.exact(0),
        )
        self._form_hint_window = Window(
            content=FormattedTextControl(self._form_hint_fragments),
            height=lambda: D.exact(1) if self._form_visible() else D.exact(0),
            style="class:screen",
        )
        self._menu_header_window = Window(
            content=FormattedTextControl(self._menu_header_fragments),
            height=lambda: D.exact(1 if self._menu_header_fragments() else 0),
            style="class:screen",
        )
        self._menu_body_window = Window(
            content=FormattedTextControl(self._menu_body_fragments),
            height=lambda: D.exact(self._menu_body_height()),
            style="class:screen",
        )
        chrome = HSplit(
            [
                self._working_window,
                Window(char="─", height=D.exact(1), style="class:chrome.rule"),
                self._input_window,
                Window(char="─", height=D.exact(1), style="class:chrome.rule"),
                self._queue_window,
                self._form_row,
                self._form_hint_window,
                self._menu_header_window,
                self._menu_body_window,
                Window(FormattedTextControl(self._stats_line), height=D.exact(1), style="class:screen"),
                Window(FormattedTextControl(self._path_line), height=D.exact(1), style="class:screen"),
            ],
            style="class:screen",
        )
        layout = Layout(HSplit([self._transcript_pane, chrome], style="class:screen"))
        layout.focus(self._input_window)
        return layout

    def _working_status_visible(self) -> bool:
        return self._working_active or bool(self._worked_label)

    def _queue_line_visible(self) -> bool:
        return bool(self.session.prompt_queue)

    def _queue_fragments(self):
        queue = self.session.prompt_queue
        if not queue:
            return []
        width = term_width()
        count = len(queue)
        head = " ".join(queue[0].strip().split())[: width - 18]
        return [
            ("class:chrome.queue", f"queued ({count}) "),
            ("class:chrome.queue.arrow", "» "),
            ("class:chrome.queue.preview", head),
        ]

    def _working_status_fragments(self):
        if self._working_active:
            return working_fragments(self._working_frame)
        if self._worked_label:
            return worked_fragments(self._worked_elapsed)
        return []

    def turn_working_active(self) -> bool:
        return self._turn_working

    def begin_turn(self) -> None:
        self._turn_working = True
        self.start_working()

    def finish_turn(self) -> None:
        self._turn_working = False
        self.stop_working()

    def start_working(self) -> None:
        if self._working_active:
            return
        self._worked_label = None
        self._worked_elapsed = 0.0
        self._working_active = True
        self._working_started_at = time.monotonic()
        self._working_frame = 0
        self._working_task = self._app.create_background_task(self._animate_working())
        self.invalidate()

    def stop_working(self) -> None:
        if self._working_task is not None and not self._working_task.done():
            self._working_task.cancel()
        self._working_task = None
        if not self._working_active:
            return
        elapsed = time.monotonic() - (self._working_started_at or time.monotonic())
        self._working_active = False
        self._working_started_at = None
        self._worked_elapsed = elapsed
        self._worked_label = f"Worked for {self._format_worked_duration(elapsed)}"
        self.invalidate()

    async def _animate_working(self) -> None:
        try:
            while self._working_active:
                await asyncio.sleep(0.08)
                if not self._working_active:
                    return
                self._working_frame += 1
                self.invalidate()
        except asyncio.CancelledError:
            return

    @staticmethod
    def _format_worked_duration(elapsed_s: float) -> str:
        if elapsed_s < 60:
            return f"{elapsed_s:.1f}s"
        minutes = int(elapsed_s // 60)
        seconds = elapsed_s % 60
        return f"{minutes}m {seconds:.1f}s"

    def _prompt_prefix(self):
        mode = self.session.agent_mode
        style = "class:prompt.mode.plan" if mode == "plan" else "class:prompt.mode"
        return [(style, f"{mode} "), ("class:prompt", "> ")]

    def _input_prefix_fragments(self):
        if self._prompt_kind == "line":
            return [("class:prompt", f"{self._prompt_title} ")]
        if self._form_kind or self._prompt_kind:
            return self._prompt_prefix() + [("class:transcript.muted", self._buffer.text or "/config")]
        return self._prompt_prefix()

    def _input_max_rows(self) -> int:
        return max(3, min(INPUT_MAX_ROWS_CAP, term_height() // INPUT_MAX_ROWS_FRACTION))

    def _input_prefix_display_width(self) -> int:
        return sum(len(s) for _, s in self._input_prefix_fragments())

    def _input_row_count(self) -> int:
        if self._form_kind or self._prompt_kind or self._menu_kind in PICKER_MODES:
            return 1
        text = self._buffer.text
        if not text:
            return 1
        prefix_w = self._input_prefix_display_width()
        wrap_w = max(10, term_width() - prefix_w)
        rows = 0
        for line in text.split("\n"):
            rows += max(1, -(-len(line) // wrap_w))
        return rows

    def _input_height(self):
        return D.exact(min(self._input_max_rows(), self._input_row_count()))

    def _form_label_display_width(self) -> int:
        return len(f"{self._visible_form_label()} :")

    def _form_field_outer_width(self) -> int:
        return FORM_FIELD_WIDTH + 2

    def _form_row_width(self) -> int:
        return self._form_label_display_width() + 1 + self._form_field_outer_width()

    def _form_row_left_pad(self) -> int:
        return max(0, (term_width() - self._form_row_width()) // 2)

    def _form_hint_left_pad(self) -> int:
        return self._form_row_left_pad() + self._form_label_display_width() + 1

    def _visible_form_label(self) -> str:
        if self._prompt_kind == "question":
            return "note"
        return self._form_label

    def _form_label_fragments(self):
        if not self._form_visible():
            return []
        return [("class:chrome.form.label", f"{self._visible_form_label()} :")]

    def _form_field_prefix_fragments(self):
        return [("class:chrome.input.box", "| "), ("class:chrome.input.field", "")]

    def _form_field_suffix_fragments(self):
        pad = max(0, FORM_FIELD_WIDTH - len(self._form_buffer.text))
        return [("class:chrome.input.box", " " * pad + "|")]

    def _form_hint_fragments(self):
        if not self._form_visible():
            return []
        hint = "Enter to submit - Tab returns to options" if self._prompt_kind == "question" else "Enter to save - Esc to cancel"
        return [("class:chrome.form.hint", (" " * self._form_hint_left_pad()) + hint)]

    def _stats_line(self):
        width = term_width()
        used = int(self.session.context_used or 0)
        total = int(self.session.context_total or 0)
        cache_key = (
            width,
            cost_tracker.input_tokens,
            cost_tracker.output_tokens,
            cost_tracker.cost_usd,
            used,
            total,
        )
        if cache_key == self._stats_line_cache_key:
            return self._stats_line_cache
        bar = context_bar(used, total)
        cost = f"${cost_tracker.cost_usd:.4f}" if cost_tracker.cost_usd > 0 else "$0.0000"
        left = f"↑ {cost_tracker.input_tokens:,}  ↓ {cost_tracker.output_tokens:,}  {cost}"
        right = f"context {used // 1000}k/{total // 1000}k used {bar}"
        gap = max(1, width - len(left) - len(right))
        fragments = [
            ("class:chrome.stats.key", "↑ "),
            ("class:chrome.stats.value", f"{cost_tracker.input_tokens:,}"),
            ("class:chrome.stats.key", "  ↓ "),
            ("class:chrome.stats.value", f"{cost_tracker.output_tokens:,}  {cost}"),
            ("class:chrome.stats.key", " " * gap),
            ("class:chrome.stats.key", "context "),
            ("class:chrome.stats.value", f"{used // 1000}k/{total // 1000}k used "),
            ("class:chrome.stats.accent", bar),
        ]
        self._stats_line_cache_key = cache_key
        self._stats_line_cache = fragments
        return fragments

    def _refresh_cwd_line_if_changed(self) -> bool:
        current = display_cwd()
        if current == self._cwd_line:
            return False
        self._cwd_line = current
        self._path_line_cache_key = None
        return True

    def _path_line(self):
        width = term_width()
        right = f"{model_footer_name(active_model_name())} - {active_reasoning_effort()}"
        cache_key = (width, self._cwd_line, right)
        if cache_key == self._path_line_cache_key:
            return self._path_line_cache
        gap = max(1, width - len(self._cwd_line) - len(right))
        fragments = [("class:chrome.path", self._cwd_line + (" " * gap) + right)]
        self._path_line_cache_key = cache_key
        self._path_line_cache = fragments
        return fragments
