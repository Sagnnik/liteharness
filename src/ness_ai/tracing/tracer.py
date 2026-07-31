from __future__ import annotations

import contextvars
import time
from typing import Any, Literal, Protocol

from ness_ai.tracing.config import TracingConfig

# --- parent span context ------------------------------------------------
# Ness AI spans are linked into a tree through this ContextVar. Every
# span's ``__enter__`` installs itself, ``__exit__`` pops it. Exporter
# backends (OTel) read the current span to derive a parent context.
_current_span: "contextvars.ContextVar[Span | None]" = contextvars.ContextVar(
    "ness_ai_current_span", default=None
)


# --- Span protocol ------------------------------------------------------
class Span(Protocol):
    """Minimal span surface every tracer backend must implement.

    Callers always use spans as context managers (``with tracer.start_span(...)
    as span``). Implementations are responsible for installing themselves in
    ``_current_span`` on enter and restoring the previous value on exit so the
    parent linkage is preserved across asynchronous boundaries.
    """

    def set_attribute(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None: ...
    def record_exception(self, exc: Exception, attributes: dict[str, Any] | None = None) -> None: ...
    def set_status(self, status: Literal["OK", "ERROR"], description: str | None = None) -> None: ...
    def end(self) -> None: ...
    def __enter__(self) -> "Span": ...
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


# --- In-memory span -----------------------------------------------------
class InMemorySpan:
    """Reference implementation used by tests and the console exporter.

    Records every attribute/event in memory and tracks duration. No network
    or thread activity is performed — ideal for assertions and lightweight
    local debugging.
    """

    def __init__(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        self.name = name
        self._attrs: dict[str, Any] = dict(attrs or {})
        self._events: list[dict[str, Any]] = []
        self._status: Literal["OK", "ERROR"] | None = None
        self._status_desc: str | None = None
        self._start = time.monotonic()
        self._ended = False
        self._token: contextvars.Token[Span | None] | None = None

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
        self._attrs["duration_ms"] = int((time.monotonic() - self._start) * 1000)
        if self._status:
            self._attrs["status"] = self._status
            if self._status_desc:
                self._attrs["status_description"] = self._status_desc

    def __enter__(self) -> "InMemorySpan":
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        self.end()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None

    @property
    def attributes(self) -> dict[str, Any]:
        return self._attrs

    @property
    def events(self) -> list[dict[str, Any]]:
        return self._events

    @property
    def status(self) -> Literal["OK", "ERROR"] | None:
        return self._status

    @property
    def ended(self) -> bool:
        return self._ended


# --- No-op span/tracer --------------------------------------------------
class NoopSpan:
    """Zero-overhead span returned by :class:`NoopTracer`."""

    def __init__(self, name: str = "") -> None:
        self.name = name

    def set_attribute(self, key: str, value: Any) -> None: ...
    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None: ...
    def record_exception(self, exc: Exception, attributes: dict[str, Any] | None = None) -> None: ...
    def set_status(self, status: Literal["OK", "ERROR"], description: str | None = None) -> None: ...
    def end(self) -> None: ...
    def __enter__(self) -> "NoopSpan":
        return self
    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class NoopTracer:
    """Default tracer. Builds :class:`NoopSpan` instances that do nothing."""

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span:
        _ = kind
        return NoopSpan(name)  # type: ignore[return-value]


# --- Tracer protocol ----------------------------------------------------
class Tracer(Protocol):
    """Backend-agnostic tracer surface.

    ``kind`` is one of :data:`KIND_INTERNAL` / :data:`KIND_CLIENT` from
    :mod:`ness_ai.tracing.semconv` (or ``None`` → internal).
    """

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span: ...


# --- Multi tracer (Fan-out classes)-------------------------------------------------------
class MultiSpan:
    """Fan-out span wrapping several backends.

    Important: ``__enter__`` installs ``self`` in ``_current_span`` exactly
    once (not once per child) so nested SDK spans attach to the composite,
    not just to whichever child happened to enter last.
    """

    def __init__(self, spans: list[Span]) -> None:
        self._spans = spans
        self._token: contextvars.Token[Span | None] | None = None

    def set_attribute(self, key: str, value: Any) -> None:
        for span in self._spans:
            span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        for span in self._spans:
            span.add_event(name, attributes)

    def record_exception(self, exc: Exception, attributes: dict[str, Any] | None = None) -> None:
        for span in self._spans:
            span.record_exception(exc, attributes)

    def set_status(self, status: Literal["OK", "ERROR"], description: str | None = None) -> None:
        for span in self._spans:
            span.set_status(status, description)

    def end(self) -> None:
        for span in self._spans:
            span.end()

    def __enter__(self) -> "MultiSpan":
        # Install ourselves once — children must NOT each push the contextvar
        # or nested spans will reattach to a single child backend.
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        self.end()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None


class MultiTracer:
    """Tracer that fans every span out to a list of backends."""

    def __init__(self, tracers: list[Tracer]) -> None:
        self._tracers = list(tracers)

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span:
        return MultiSpan([t.start_span(name, attributes, kind) for t in self._tracers])


# --- factory ------------------------------------------------------------
def build_tracer(config: TracingConfig | None = None) -> Tracer:
    """Build a tracer from a :class:`TracingConfig`.

    Returns :class:`NoopTracer` when tracing is disabled or the exporter is
    ``"none"``. OTel/console exporters are imported lazily so the optional
    ``opentelemetry`` dependency is only required when actually used.
    """
    if config is None or not config.enabled:
        return NoopTracer()
    if config.exporter == "otlp":
        try:
            from ness_ai.tracing.exporters.otlp import OTelTracer
        except ImportError as exc:  # pragma: no cover - exercised when extras missing
            raise ImportError(
                "OTLP tracing requires the 'tracing' extra. "
                "Install it with: pip install 'ness-ai[tracing]'"
            ) from exc
        return OTelTracer(config)
    if config.exporter == "console":
        from ness_ai.tracing.exporters.console import ConsoleTracer
        return ConsoleTracer()
    return NoopTracer()