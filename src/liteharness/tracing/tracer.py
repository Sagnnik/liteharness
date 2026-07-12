from __future__ import annotations

from typing import Any, Protocol

from liteharness.tracing.config import TracingConfig


class Span:
    """No-op span used until a real OTel backend is wired."""

    def __init__(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        self.name = name
        self.attrs = dict(attrs or {})

    def set_attribute(self, key: str, value: Any) -> None:
        self.attrs[key] = value

    def __enter__(self) -> Span:
        return self

    def __exit__(self, *exc: object) -> None:
        return None


class Tracer(Protocol):
    def start_span(self, name: str, **attrs: Any) -> Span: ...


class NoopTracer:
    def start_span(self, name: str, **attrs: Any) -> Span:
        return Span(name, attrs)


def build_tracer(config: TracingConfig | None = None) -> Tracer:
    """Return a tracer. P0 always returns NoopTracer; OTel backends come later."""
    _ = config
    return NoopTracer()
