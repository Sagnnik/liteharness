"""Tests for the OTel exporter.

Skipped automatically when the optional ``opentelemetry`` extra is not
installed. When available, an :class:`InMemorySpanExporter` is attached so we
can assert that parent→child links survive the liteharness ContextVar-based
parent linkage mechanism.
"""

from __future__ import annotations

import pytest

otlp = pytest.importorskip("opentelemetry")  # noqa: F841  (gate for whole module)


from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: E402
    OTLPSpanExporter,
)
from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)

from liteharness.tracing.config import TracingConfig  # noqa: E402
from liteharness.tracing.exporters.otlp import OTelTracer  # noqa: E402


@pytest.fixture
def in_memory():
    """Provide an in-memory OTel exporter wired into a fresh OTelTracer."""
    exporter = InMemorySpanExporter()
    provider = OTelTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return exporter, provider


def _make_tracer_with_in_memory_exporter(in_memory_exporter):
    """Construct an OTelTracer but swap its span processor for the in-memory one."""
    config = TracingConfig(enabled=True, exporter="otlp", service_name="test")
    tracer = OTelTracer(config)
    # replace the provider with one wired to the in-memory exporter
    provider = OTelTracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(in_memory_exporter))
    tracer._tracer = provider.get_tracer("liteharness-test")
    tracer._provider = provider
    return tracer


def test_otel_tracer_produces_connected_trace_tree(in_memory):
    exporter, _provider = in_memory
    tracer = _make_tracer_with_in_memory_exporter(exporter)
    with tracer.start_span("parent", kind="client") as p:
        with tracer.start_span("child", kind="client") as c:
            c.set_attribute("k", "v")
    # force flush
    tracer._provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 2
    by_name = {s.name: s for s in spans}
    assert "parent" in by_name
    assert "child" in by_name
    parent_span = by_name["parent"]
    child_span = by_name["child"]
    # Child must point at parent via parent_span_id; same trace_id.
    assert child_span.parent.span_id == parent_span.context.span_id
    assert child_span.context.trace_id == parent_span.context.trace_id


def test_otel_tracer_attributes_propagate(in_memory):
    exporter, _provider = in_memory
    tracer = _make_tracer_with_in_memory_exporter(exporter)
    with tracer.start_span(
        "agent.llm_call",
        attributes={"gen_ai.request.model": "gpt-4o", "gen_ai.system": "liteharness"},
        kind="client",
    ) as s:
        s.set_attribute("gen_ai.usage.input_tokens", 100)
        s.set_attribute("gen_ai.usage.output_tokens", 10)
    tracer._provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    attrs = dict(span.attributes or {})
    assert attrs["gen_ai.request.model"] == "gpt-4o"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["gen_ai.usage.output_tokens"] == 10


def test_otel_tracer_records_exception(in_memory):
    exporter, _provider = in_memory
    tracer = _make_tracer_with_in_memory_exporter(exporter)
    with pytest.raises(RuntimeError):
        with tracer.start_span("op", kind="client") as s:
            s.set_attribute("before", "raise")
            raise RuntimeError("boom")
    tracer._provider.force_flush()
    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    span = spans[0]
    # OTel records exception as an event
    events = span.events or []
    assert any(
        ev.name == "exception"
        or (ev.attributes or {}).get("exception.message") == "boom"
        for ev in events
    )
    # Status must be ERROR
    from opentelemetry.trace.status import StatusCode
    assert span.status.status_code == StatusCode.ERROR