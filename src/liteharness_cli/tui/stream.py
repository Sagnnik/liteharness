from __future__ import annotations

import time
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from liteharness_cli.tui.app import TuiApp

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


def _active_sink():
    """Lazy accessor for ``render._sink()``.

    Imported lazily so ``liteharness_cli.tui.stream`` does not depend on ``liteharness_cli.tui.render`` at
    module load time (``render`` itself imports ``liteharness_cli.tui.stream`` for the
    ``AssistantStream`` re-export). The indirection mirrors the original
    in-method ``import time`` pattern but for the sink registry.
    """
    from liteharness_cli.tui.render import _sink

    return _sink()


class AssistantStream:
    def __init__(self) -> None:
        self._stream = _active_sink().start_assistant_stream()
        self._buffer: list[str] = []
        # Per-LLM-call reasoning state. The facade owns it (not the turn
        # driver) so the live assistant-stream inner object and the reasoning
        # slot sit next to each other and the cancel path can drain partial
        # reasoning via ``reasoning_state`` before ``stop()`` finalises the
        # assistant text.
        self._reasoning: list[str] = []
        self._reasoning_started_at: float | None = None
        self._reasoning_slot: dict | None = None

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer.append(chunk)
        self._stream.feed(chunk)

    def feed_reasoning(self, chunk: str) -> None:
        # OpenRouter streams the chain-of-thought on ``additional_kwargs``
        # separately from ``content``; pure-reasoning chunks arrive here with
        # an empty content string. The slot is reserved lazily on the first
        # fragment so the reasoning block sits above the assistant stream's
        # reserved markdown slot (Anthropic/OpenCode convention). The inner
        # TuiAssistantStream's ``_line_start`` is shifted by the insert so the
        # finalised assistant text writes into the correct transcript slice.
        if not chunk:
            return
        import time as _time

        if self._reasoning_started_at is None:
            self._reasoning_started_at = _time.monotonic()
        if self._reasoning_slot is None:
            # The active sink's stream (TuiAssistantStream) exposes
            # ``_line_start`` for live-transcript shifting; duck-type via
            # hasattr so render.py does not import the concrete type and
            # stays free of circular dependencies.
            inner = self._stream if hasattr(self._stream, "_line_start") else None
            self._reasoning_slot = _active_sink().reserve_reasoning_slot(inner)
        self._reasoning.append(chunk)

    def finalize_reasoning(self) -> None:
        if not self._reasoning or self._reasoning_slot is None:
            return
        import time as _time

        elapsed = _time.monotonic() - (self._reasoning_started_at or _time.monotonic())
        _active_sink().finalize_reasoning_slot(
            self._reasoning_slot, "".join(self._reasoning), elapsed=elapsed
        )
        self._reasoning = []
        self._reasoning_started_at = None
        self._reasoning_slot = None

    def reasoning_state(self) -> tuple[str | None, float]:
        """Return ``(text, elapsed)`` for the in-flight reasoning buffer.

        Used by the cancel-finalize path to render partial reasoning before
        ``stop()`` discards live-stream state. ``text`` is ``None`` when no
        reasoning was captured this LLM call.
        """
        if not self._reasoning:
            return None, 0.0
        import time as _time

        text = "".join(self._reasoning)
        elapsed = _time.monotonic() - (self._reasoning_started_at or _time.monotonic())
        return text, elapsed

    @property
    def text(self) -> str:
        return "".join(self._buffer)

    def stop(self) -> None:
        self._stream.stop()
