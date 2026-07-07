from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cli.app import TuiApp

STREAM_FLUSH_INTERVAL_S = 0.05
STREAM_FLUSH_CHARS = 256


class TuiAssistantStream:
    def __init__(self, ui: TuiApp) -> None:
        self.ui = ui
        self._buffer: list[str] = []
        self._line_start: int | None = None
        self._line_count = 0
        self._last_flush = 0.0
        self._total_chars = 0
        self._flushed_chars = 0
        self.flush_count = 0

    def shift_start(self, delta: int) -> None:
        """Shift the reserved transcript slot by ``delta`` lines.

        Called when a reasoning block is inserted above this stream's slot
        (the "thinking block above assistant" UX convention). Bumps
        ``_line_start`` so the final ``finalize_assistant_stream`` writes
        the assistant markdown into the right place.
        """
        if self._line_start is not None:
            self._line_start += delta

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer.append(chunk)
        self._total_chars += len(chunk)
        if self._should_flush(chunk):
            self._flush()

    def _should_flush(self, chunk: str) -> bool:
        if self._line_start is None:
            return True
        if "\n" in chunk:
            return True
        if self._total_chars - self._flushed_chars >= STREAM_FLUSH_CHARS:
            return True
        return time.monotonic() - self._last_flush >= STREAM_FLUSH_INTERVAL_S

    def _flush(self) -> None:
        text = "".join(self._buffer)
        self._line_start, self._line_count = self.ui.set_assistant_stream(
            text,
            self._line_start,
            self._line_count,
        )
        self._last_flush = time.monotonic()
        self._flushed_chars = len(text)
        self.flush_count += 1

    def stop(self) -> None:
        text = "".join(self._buffer)
        if not text.strip():
            self.ui.clear_assistant_stream(self._line_start, self._line_count)
            return
        self.ui.finalize_assistant_stream(text, self._line_start, self._line_count)


class Thinking:
    def __init__(self, ui: TuiApp, label: str = "thinking") -> None:
        self.ui = ui
        self.label = label
        self._owned = False

    def __enter__(self):
        if not self.ui.turn_working_active():
            self.ui.start_working()
            self._owned = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._owned:
            self.ui.stop_working()
        return None
