"""Console tracer backend.

Prints ``[trace]`` lines to stdout for each span lifecycle event. Useful for
debugging instrumentation without an OTel collector. Each span instance still
manages the ``_current_span`` ContextVar so a parent→child trace tree exists
even when there is only one consumer.
"""

from __future__ import annotations

import contextvars
import sys
import time
from typing import Any, IO, Literal

from liteharness.tracing.tracer import Span, _current_span


class ConsoleSpan:
    """Span implementation that prints lifecycle events to stdout/stderr."""

    def __init__(
        self,
        name: str,
        stream: IO[str],
        attributes: dict[str, Any] | None = None,
        indent: int = 0,
    ) -> None:
        self.name = name
        self._stream = stream
        self._attrs: dict[str, Any] = dict(attributes or {})
        self._events: list[dict[str, Any]] = []
        self._status: Literal["OK", "ERROR"] | None = None
        self._status_desc: str | None = None
        self._start = time.monotonic()
        self._ended = False
        self._indent = indent
        self._token: contextvars.Token[Span | None] | None = None
        prefix = "  " * indent
        self._stream.write(
            f"[trace] {prefix}BEGIN {name}"
            + (f" attrs={_fmt(self._attrs)}" if self._attrs else "")
            + "\n"
        )
        self._stream.flush()

    def set_attribute(self, key: str, value: Any) -> None:
        self._attrs[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._events.append(
            {"name": name, "attributes": dict(attributes or {}), "timestamp": time.monotonic()}
        )

    def record_exception(self, exc: Exception, attributes: dict[str, Any] | None = None) -> None:
        self.add_event(
            "exception",
            {"exception.type": type(exc).__name__, "exception.message": str(exc), **(attributes or {})},
        )
        self.set_status("ERROR", str(exc))

    def set_status(self, status: Literal["OK", "ERROR"], description: str | None = None) -> None:
        self._status = status
        self._status_desc = description

    def end(self) -> None:
        if self._ended:
            return
        self._ended = True
        duration_ms = int((time.monotonic() - self._start) * 1000)
        status_str = ""
        if self._status is not None:
            status_str = f" status={self._status}"
            if self._status_desc:
                status_str += f" ({self._status_desc})"
        extra = {k: v for k, v in self._attrs.items() if k not in ("status", "status_description", "duration_ms")}
        line = (
            f"[trace] {'  ' * self._indent}END   {self.name}"
            f" duration_ms={duration_ms}{status_str}"
            + (f" attrs={_fmt(extra)}" if extra else "")
            + (f" events={_fmt(self._events)}" if self._events else "")
            + "\n"
        )
        self._stream.write(line)
        self._stream.flush()

    def __enter__(self) -> ConsoleSpan:
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        self.end()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None


def _fmt(obj: Any) -> str:
    try:
        return str(obj)
    except Exception:  # pragma: no cover - defensive
        return repr(obj)


class ConsoleTracer:
    """Builds :class:`ConsoleSpan` instances that print to ``stream``."""

    def __init__(self, stream: IO[str] | None = None) -> None:
        self._stream = stream or sys.stdout

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span:
        _ = kind
        # compute indentation depth from the ContextVar chain
        depth = 0
        cur = _current_span.get()
        while isinstance(cur, ConsoleSpan):
            depth += 1
            cur = None  # only count one level
        return ConsoleSpan(name, self._stream, attributes, indent=depth)  # type: ignore[return-value]