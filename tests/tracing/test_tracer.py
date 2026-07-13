"""Tests for liteharness.tracing.tracer primitives."""

from __future__ import annotations

import pytest

from liteharness.tracing.tracer import (
    InMemorySpan,
    MultiSpan,
    MultiTracer,
    NoopSpan,
    NoopTracer,
    _current_span,
)


def test_noop_tracer_returns_noop_span():
    tracer = NoopTracer()
    span = tracer.start_span("x", attributes={"a": 1})
    assert isinstance(span, NoopSpan)
    with span as entered:
        assert entered is span
        entered.set_attribute("k", "v")
        entered.add_event("e", {"x": 1})
        entered.record_exception(RuntimeError("boom"))
        entered.set_status("ERROR", "boom")
        entered.end()


def test_in_memory_span_records_attrs_events():
    span = InMemorySpan("op", {"k0": "v0"})
    span.set_attribute("k1", 1)
    span.add_event("event1", {"attr": "val"})
    span.record_exception(ValueError("bad"))
    span.set_status("ERROR", "bad")
    span.end()
    assert span.attributes["k0"] == "v0"
    assert span.attributes["k1"] == 1
    assert span.attributes["status"] == "ERROR"
    assert span.attributes["status_description"] == "bad"
    assert "duration_ms" in span.attributes
    assert any(ev["name"] == "event1" for ev in span.events)
    exc_events = [ev for ev in span.events if ev["name"] == "exception"]
    assert exc_events and exc_events[0]["attributes"]["exception.message"] == "bad"
    assert span.ended


def test_in_memory_span_enter_installs_and_resets_contextvar():
    assert _current_span.get() is None
    with InMemorySpan("outer") as outer:
        assert _current_span.get() is outer
    assert _current_span.get() is None


def test_in_memory_span_exit_with_exception_records_error():
    span = InMemorySpan("op")
    with pytest.raises(RuntimeError):
        with span:
            raise RuntimeError("boom")
    assert span.ended
    assert span.attributes["status"] == "ERROR"
    exc_events = [ev for ev in span.events if ev["name"] == "exception"]
    assert exc_events


def test_multitracer_fans_out_to_all_backends():
    captured_a: list[str] = []
    captured_b: list[str] = []

    class _TracerA:
        def start_span(self, name, attributes=None, kind=None):
            captured_a.append(name)
            return InMemorySpan(name, attributes)

    class _TracerB:
        def start_span(self, name, attributes=None, kind=None):
            captured_b.append(name)
            return InMemorySpan(name, attributes)

    multi = MultiTracer([_TracerA(), _TracerB()])  # type: ignore[arg-type]
    with multi.start_span("hello", attributes={"k": "v"}) as span:
        assert isinstance(span, MultiSpan)
        assert _current_span.get() is span
    assert captured_a == ["hello"]
    assert captured_b == ["hello"]
    assert _current_span.get() is None


def test_multispan_does_not_install_each_child_in_contextvar():
    """MultiSpan must install itself once, not once per child backend."""
    child = InMemorySpan("inner")
    span = MultiSpan([child])  # type: ignore[list-item]
    with span:
        assert _current_span.get() is span
    # Single reset restores None — not a stack of resets per child.
    assert _current_span.get() is None


def test_multispan_records_exception_on_exit():
    child = InMemorySpan("inner")
    span = MultiSpan([child])  # type: ignore[list-item]
    with pytest.raises(RuntimeError):
        with span:
            raise RuntimeError("kaboom")
    # the exception should have been recorded on the inner span
    exc_events = [ev for ev in child.events if ev["name"] == "exception"]
    assert exc_events


def test_noop_span_idempotent_end():
    span = NoopSpan()
    span.end()
    span.end()  # second call should be a no-op


def test_build_tracer_returns_noop_when_disabled():
    from liteharness.tracing.config import TracingConfig
    from liteharness.tracing.tracer import build_tracer

    assert isinstance(build_tracer(None), NoopTracer)
    assert isinstance(build_tracer(TracingConfig()), NoopTracer)
    assert isinstance(
        build_tracer(TracingConfig(enabled=True, exporter="none")), NoopTracer
    )


def test_build_tracer_console_does_not_require_opentelemetry():
    from liteharness.tracing.config import TracingConfig
    from liteharness.tracing.exporters.console import ConsoleTracer
    from liteharness.tracing.tracer import build_tracer

    tracer = build_tracer(TracingConfig(enabled=True, exporter="console"))
    assert isinstance(tracer, ConsoleTracer)


def test_build_tracer_otlp_raises_friendly_importerror_without_extra():
    """Without the 'tracing' extra installed, OTLP must give a clear message."""
    from liteharness.tracing.config import TracingConfig
    from liteharness.tracing.tracer import build_tracer

    try:
        import opentelemetry  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError) as excinfo:
            build_tracer(TracingConfig(enabled=True, exporter="otlp"))
        assert "tracing" in str(excinfo.value).lower()
    else:  # opentelemetry IS installed in the environment — just skip
        pytest.skip("opentelemetry is installed; ImportError path cannot be tested")