from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Callable

from prompt_toolkit.data_structures import Point
from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import UIContent, UIControl
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from ness_cli.tui.constants import MOUSE_SCROLL_LINES
from ness_cli.tui.models import TranscriptLine

DEFAULT_TRANSCRIPT_WIDTH = 80
SELECTION_STYLE = "class:transcript.selection"


@dataclass(frozen=True)
class VisualPosition:
    row: int
    col: int


@dataclass(eq=False)
class TranscriptBlock:
    """Stable handle for a contiguous, store-owned transcript block.

    ``start`` is maintained by :class:`TranscriptStore` whenever another
    block is inserted, removed, or changes size.  Callers may retain the
    handle across those mutations instead of caching a fragile line index.
    """

    start: int
    count: int
    attached: bool = True


class TranscriptStore:
    """Transcript source lines plus cached visual row offsets."""

    def __init__(self, lines: list[TranscriptLine], width: int = DEFAULT_TRANSCRIPT_WIDTH) -> None:
        self.lines = lines
        self.width = max(1, width)
        self.revision = 0
        self.has_user_blocks = any(line.user_source for line in lines)
        self._plain_text: str | None = None
        self._row_counts: list[int] = []
        self._row_offsets: list[int] = [0]
        self._blocks: list[TranscriptBlock] = []
        self._rebuild_rows()

    def set_width(self, width: int) -> bool:
        width = max(1, width)
        if width == self.width:
            return False
        self.width = width
        self._rebuild_rows()
        return True

    def append(self, lines: list[TranscriptLine]) -> None:
        if not lines:
            return
        self.lines.extend(lines)
        for line in lines:
            row_count = self._row_count(line)
            self._row_counts.append(row_count)
            self._row_offsets.append(self._row_offsets[-1] + row_count)
        self._mark_changed(lines)

    def replace(self, start: int, count: int, lines: list[TranscriptLine]) -> None:
        start, count = self._validate_untracked_mutation(start, count)
        self._shift_blocks_for_range(start, count, len(lines))
        self.lines[start : start + count] = lines
        self._row_counts[start : start + count] = [self._row_count(line) for line in lines]
        self._rebuild_offsets_from(start)
        self._mark_changed()

    def delete(self, start: int, count: int) -> None:
        if count <= 0:
            return
        start, count = self._validate_untracked_mutation(start, count)
        self._shift_blocks_for_range(start, count, 0)
        del self.lines[start : start + count]
        del self._row_counts[start : start + count]
        self._rebuild_offsets_from(start)
        self._mark_changed()

    def insert(self, start: int, lines: list[TranscriptLine]) -> None:
        """Insert ``lines`` before index ``start``.

        Symmetric to ``delete``: row offsets are rebuilt from the touched
        index onward and stable block handles at or after the insertion are
        shifted automatically.
        """
        if not lines:
            return
        start = max(0, min(start, len(self.lines)))
        for block in self._blocks:
            if block.start < start < block.start + block.count:
                raise ValueError(
                    "raw transcript insertion splits a tracked block; "
                    "use the tracked-block operation"
                )
        self._shift_blocks_for_range(start, 0, len(lines))
        self.lines[start:start] = lines
        self._row_counts[start:start] = [self._row_count(line) for line in lines]
        self._rebuild_offsets_from(start)
        self._mark_changed(lines)

    def append_tracked(self, lines: list[TranscriptLine]) -> TranscriptBlock:
        """Append ``lines`` and return a stable handle for the new block."""
        start = len(self.lines)
        self.append(lines)
        return self._register_block(start, len(lines))

    def insert_tracked(
        self, start: int, lines: list[TranscriptLine]
    ) -> TranscriptBlock:
        """Insert a tracked block before ``start`` and return its handle."""
        start = max(0, min(start, len(self.lines)))
        self.insert(start, lines)
        return self._register_block(start, len(lines))

    def replace_tracked(
        self, block: TranscriptBlock, lines: list[TranscriptLine]
    ) -> None:
        """Replace a tracked block while preserving its stable identity."""
        self._require_attached(block)
        start = block.start
        old_count = block.count
        delta = len(lines) - old_count
        old_end = start + old_count
        for other in self._blocks:
            if other is not block and other.start >= old_end:
                other.start += delta
        self.lines[start:old_end] = lines
        self._row_counts[start:old_end] = [self._row_count(line) for line in lines]
        block.count = len(lines)
        self._rebuild_offsets_from(start)
        self._mark_changed()

    def delete_tracked(self, block: TranscriptBlock) -> None:
        """Delete a tracked block and detach its handle."""
        self._require_attached(block)
        start = block.start
        count = block.count
        end = start + count
        del self.lines[start:end]
        del self._row_counts[start:end]
        for other in self._blocks:
            if other is not block and other.start >= end:
                other.start -= count
        self._detach_block(block)
        self._rebuild_offsets_from(start)
        self._mark_changed()

    def release_tracked(self, block: TranscriptBlock) -> None:
        """Stop tracking ``block`` without removing its transcript lines."""
        self._require_attached(block)
        self._detach_block(block)

    def move_tracked_to_end(self, block: TranscriptBlock) -> None:
        """Move a tracked block after every other transcript line.

        The handle remains attached and every block crossed by the move has
        its start adjusted.  This is useful for content whose final ordering
        is known only after later streaming events arrive.
        """
        self._require_attached(block)
        start = block.start
        count = block.count
        end = start + count
        if end == len(self.lines):
            return

        moved_lines = self.lines[start:end]
        moved_row_counts = self._row_counts[start:end]
        del self.lines[start:end]
        del self._row_counts[start:end]
        for other in self._blocks:
            if other is not block and other.start >= end:
                other.start -= count
        block.start = len(self.lines)
        self.lines.extend(moved_lines)
        self._row_counts.extend(moved_row_counts)
        self._rebuild_offsets_from(start)
        self._mark_changed()

    def reset(self, lines: list[TranscriptLine] | None = None) -> None:
        for block in self._blocks:
            block.attached = False
        self._blocks.clear()
        if lines is not None:
            self.lines[:] = lines
        self._rebuild_rows()
        self._mark_changed()

    def _register_block(self, start: int, count: int) -> TranscriptBlock:
        end = start + count
        for block in self._blocks:
            block_end = block.start + block.count
            if start < block_end and block.start < end:
                raise ValueError("tracked transcript blocks cannot overlap")
        block = TranscriptBlock(start=start, count=count)
        self._blocks.append(block)
        return block

    def _require_attached(self, block: TranscriptBlock) -> None:
        if not block.attached or block not in self._blocks:
            raise ValueError("transcript block is detached")

    def _detach_block(self, block: TranscriptBlock) -> None:
        self._blocks.remove(block)
        block.attached = False

    def _validate_untracked_mutation(self, start: int, count: int) -> tuple[int, int]:
        start = max(0, min(start, len(self.lines)))
        count = max(0, min(count, len(self.lines) - start))
        end = start + count
        for block in self._blocks:
            block_end = block.start + block.count
            if count == 0:
                continue
            if start < block_end and block.start < end:
                raise ValueError(
                    "raw transcript mutation overlaps a tracked block; "
                    "use the tracked-block operation"
                )
        return start, count

    def _shift_blocks_for_range(
        self, start: int, old_count: int, new_count: int
    ) -> None:
        old_end = start + old_count
        delta = new_count - old_count
        for block in self._blocks:
            if block.start >= old_end:
                block.start += delta

    @property
    def total_rows(self) -> int:
        return self._row_offsets[-1] if self._row_offsets else 0

    def max_scroll(self, viewport_rows: int) -> int:
        return max(0, self.total_rows - max(1, viewport_rows))

    def plain_text(self) -> str:
        if self._plain_text is None:
            self._plain_text = "".join(line.text + "\n" for line in self.lines)
        return self._plain_text

    def row_text(self, row: int) -> str:
        line_index, offset = self._line_for_row(row)
        if line_index is None:
            return ""
        line = self.lines[line_index]
        start = offset * self.width
        end = start + self.width
        return line.text[start:end]

    def row_fragments(
        self,
        row: int,
        selection: tuple[VisualPosition, VisualPosition] | None = None,
    ) -> StyleAndTextTuples:
        line_index, offset = self._line_for_row(row)
        if line_index is None:
            return []
        line = self.lines[line_index]
        start = offset * self.width
        end = start + self.width
        fragments = self._slice_fragments(line, start, end)
        if selection is None:
            return fragments
        selected = self._selected_columns(row, len(line.text[start:end]), selection)
        if selected is None:
            return fragments
        return self._apply_selection(fragments, selected[0], selected[1])

    def copy_range(self, start: VisualPosition, end: VisualPosition) -> str:
        start, end = self._ordered(start, end)
        rows: list[str] = []
        for row in range(start.row, end.row + 1):
            text = self.row_text(row)
            left = start.col if row == start.row else 0
            right = end.col if row == end.row else len(text)
            left = max(0, min(len(text), left))
            right = max(0, min(len(text), right))
            rows.append(text[left:right])
        return "\n".join(rows)

    def _rebuild_rows(self) -> None:
        self._row_counts = [self._row_count(line) for line in self.lines]
        self._row_offsets = [0] * (len(self._row_counts) + 1)
        self._rebuild_offsets_from(0)

    def _rebuild_offsets_from(self, start: int) -> None:
        if len(self._row_offsets) != len(self._row_counts) + 1:
            self._row_offsets = [0] * (len(self._row_counts) + 1)
            start = 0
        start = max(0, min(start, len(self._row_counts)))
        if start == 0:
            self._row_offsets[0] = 0
        for index in range(start, len(self._row_counts)):
            self._row_offsets[index + 1] = self._row_offsets[index] + self._row_counts[index]

    def _row_count(self, line: TranscriptLine) -> int:
        length = len(line.text)
        return max(1, (length + self.width - 1) // self.width)

    def _mark_changed(self, lines: list[TranscriptLine] | None = None) -> None:
        self.revision += 1
        self._plain_text = None
        if lines is None:
            self.has_user_blocks = any(line.user_source for line in self.lines)
        elif not self.has_user_blocks:
            self.has_user_blocks = any(line.user_source for line in lines)

    def _line_for_row(self, row: int) -> tuple[int | None, int]:
        if row < 0 or row >= self.total_rows or not self.lines:
            return None, 0
        index = max(0, bisect_right(self._row_offsets, row) - 1)
        return index, row - self._row_offsets[index]

    def _line_fragments(self, line: TranscriptLine) -> StyleAndTextTuples:
        if line.fragments and "".join(text for _, text in line.fragments) == line.text:
            return line.fragments
        return [(line.style or "class:transcript.muted", line.text)]

    def _slice_fragments(self, line: TranscriptLine, start: int, end: int) -> StyleAndTextTuples:
        row: StyleAndTextTuples = []
        cursor = 0
        for style, text in self._line_fragments(line):
            next_cursor = cursor + len(text)
            if next_cursor > start and cursor < end:
                part_start = max(0, start - cursor)
                part_end = min(len(text), end - cursor)
                row.append((style, text[part_start:part_end]))
            cursor = next_cursor
        if row:
            return row
        return [(line.style or "class:transcript.muted", "")]

    def _selected_columns(
        self,
        row: int,
        row_len: int,
        selection: tuple[VisualPosition, VisualPosition],
    ) -> tuple[int, int] | None:
        start, end = self._ordered(*selection)
        if row < start.row or row > end.row:
            return None
        left = start.col if row == start.row else 0
        right = end.col if row == end.row else row_len
        left = max(0, min(row_len, left))
        right = max(0, min(row_len, right))
        if right <= left:
            return None
        return left, right

    def _apply_selection(self, fragments: StyleAndTextTuples, left: int, right: int) -> StyleAndTextTuples:
        out: StyleAndTextTuples = []
        cursor = 0
        for style, text in fragments:
            next_cursor = cursor + len(text)
            if next_cursor <= left or cursor >= right:
                out.append((style, text))
            else:
                before = max(0, left - cursor)
                after = max(0, next_cursor - right)
                selected_start = before
                selected_end = len(text) - after
                if before:
                    out.append((style, text[:before]))
                out.append((f"{style} {SELECTION_STYLE}".strip(), text[selected_start:selected_end]))
                if after:
                    out.append((style, text[selected_end:]))
            cursor = next_cursor
        return out

    @staticmethod
    def _ordered(start: VisualPosition, end: VisualPosition) -> tuple[VisualPosition, VisualPosition]:
        if (end.row, end.col) < (start.row, start.col):
            return end, start
        return start, end


class TranscriptViewportControl(UIControl):
    """Virtualized transcript renderer with mouse scroll and selection."""

    def __init__(
        self,
        store: TranscriptStore,
        *,
        get_scroll: Callable[[], int],
        set_scroll: Callable[[int], None],
        on_scroll: Callable[[int], None],
        on_render_size: Callable[[int, int], None],
        focus: Callable[[], None],
        invalidate: Callable[[], None],
    ) -> None:
        self.store = store
        self._get_scroll = get_scroll
        self._set_scroll = set_scroll
        self._on_scroll = on_scroll
        self._on_render_size = on_render_size
        self._focus = focus
        self._invalidate = invalidate
        self._selection_anchor: VisualPosition | None = None
        self._selection_cursor: VisualPosition | None = None
        self._selecting = False

    def is_focusable(self) -> bool:
        return True

    def preferred_height(self, width: int, max_available_height: int, wrap_lines, get_line_prefix) -> int:
        return self.store.total_rows

    def create_content(self, width: int, height: int) -> UIContent:
        width = max(1, width)
        self._on_render_size(width, max(1, height))
        line_count = max(1, self.store.total_rows)
        scroll = max(0, min(self._get_scroll(), max(0, line_count - max(1, height))))
        self._set_scroll(scroll)
        selection = self.selection_range()

        def get_line(row: int) -> StyleAndTextTuples:
            return self.store.row_fragments(row, selection=selection)

        return UIContent(
            get_line=get_line,
            line_count=line_count,
            cursor_position=Point(x=0, y=min(scroll, line_count - 1)),
            show_cursor=False,
        )

    def mouse_handler(self, mouse_event: MouseEvent):
        event_type = mouse_event.event_type
        if event_type == MouseEventType.SCROLL_UP:
            self._on_scroll(-MOUSE_SCROLL_LINES)
            self._invalidate()
            return None
        if event_type == MouseEventType.SCROLL_DOWN:
            self._on_scroll(MOUSE_SCROLL_LINES)
            self._invalidate()
            return None
        if event_type == MouseEventType.MOUSE_DOWN:
            self._focus()
            position = self._event_position(mouse_event)
            self._selection_anchor = position
            self._selection_cursor = position
            self._selecting = True
            self._invalidate()
            return None
        if event_type == MouseEventType.MOUSE_MOVE and self._selecting:
            self._selection_cursor = self._event_position(mouse_event)
            self._invalidate()
            return None
        if event_type == MouseEventType.MOUSE_UP and self._selecting:
            self._selection_cursor = self._event_position(mouse_event)
            self._selecting = False
            self._invalidate()
            return None
        return NotImplemented

    def has_selection(self) -> bool:
        selection = self.selection_range()
        return selection is not None and selection[0] != selection[1]

    def selection_range(self) -> tuple[VisualPosition, VisualPosition] | None:
        if self._selection_anchor is None or self._selection_cursor is None:
            return None
        return self._selection_anchor, self._selection_cursor

    def selected_text(self) -> str:
        selection = self.selection_range()
        if selection is None or selection[0] == selection[1]:
            return ""
        return self.store.copy_range(*selection)

    def clear_selection(self) -> None:
        self._selection_anchor = None
        self._selection_cursor = None
        self._selecting = False

    def _event_position(self, mouse_event: MouseEvent) -> VisualPosition:
        max_row = max(0, self.store.total_rows - 1)
        row = max(0, min(max_row, mouse_event.position.y))
        text = self.store.row_text(row)
        col = max(0, min(len(text), mouse_event.position.x))
        return VisualPosition(row, col)
