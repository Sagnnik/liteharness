"""OpenTelemetry tracer backend.

Constructs a ``TracerProvider`` with a :class:`BatchSpanProcessor` driving an
:class:`OTLPSpanExporter` (HTTP/protobuf by default). Every span is wrapped in
an :class:`OTelSpan` that adapts the ness_ai :class:`Span` Protocol to the
OpenTelemetry surface and manages our own ``_current_span`` ContextVar so
instrumentation call sites do not need to thread OTel contexts manually.

Imports of ``opentelemetry`` are deferred until this module is first imported
by :func:`ness_ai.tracing.tracer.build_tracer` — see
``pyproject.toml`` ``[project.optional-dependencies] tracing``.
"""

from __future__ import annotations

import contextvars
from typing import Any, Literal

from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider as OTelTracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.trace import SpanKind as OTelSpanKind
from opentelemetry.trace.status import Status, StatusCode

from ness_ai.tracing.config import TracingConfig
from ness_ai.tracing.tracer import Span, _current_span

# helps in rendering different types of spans in the UI
_KIND_MAP: dict[str, OTelSpanKind] = {
    "internal": OTelSpanKind.INTERNAL, # inside the agent
    "client": OTelSpanKind.CLIENT, # outside calls like llm calls
}

# otel's own status enum to map our status to it
def _otel_status_code(status: Literal["OK", "ERROR"]) -> StatusCode:
    return StatusCode.OK if status == "OK" else StatusCode.ERROR

# bridge between ness_ai span protocol and otel span
class OTelSpan:
    """Adapts an OTel span to the ness_ai :class:`Span` Protocol.

    The OTel span itself does NOT manage a contextvar; this class installs
    itself in ``_current_span`` on ``__enter__`` so downstream ness_ai
    spans derive their parent link via :class:`OTelTracer.start_span`.
    """

    def __init__(self, span: trace.Span, capture_tool_args: bool = False) -> None:
        self._span = span # otel span object
        self._capture_tool_args = capture_tool_args
        self._token: contextvars.Token[Span | None] | None = None # for contextvar restore

    # The OTel span is exposed so OTelTracer can derive a parent context
    # self._span.{method} wraps an OTel trace.Span internally
    @property
    def _otel_span(self) -> trace.Span:
        return self._span

    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        self._span.add_event(name, attributes or {})

    def record_exception(self, exc: Exception, attributes: dict[str, Any] | None = None) -> None:
        self._span.record_exception(exc, attributes)
        self.set_status("ERROR", str(exc))

    def set_status(self, status: Literal["OK", "ERROR"], description: str | None = None) -> None:
        self._span.set_status(Status(_otel_status_code(status), description or ""))

    def end(self) -> None:
        self._span.end()

    def __enter__(self) -> OTelSpan:
        self._token = _current_span.set(self)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if exc_val is not None and isinstance(exc_val, Exception):
            self.record_exception(exc_val)
        self.end()
        if self._token is not None:
            _current_span.reset(self._token)
            self._token = None


class OTelTracer:
    """Builds :class:`OTelSpan` instances on top of an OTel ``TracerProvider``."""

    def __init__(self, config: TracingConfig) -> None:
        # Identify the this app for the backend (grafana, jeager, etc.)
        resource = Resource.create(
            {"service.name": config.service_name, **config.resource_attrs}
        )
        
        # create the OTel's built-in OTelTracerProvider object. Need to create one per process
        provider = OTelTracerProvider(resource=resource)
        
        # create the HTTP exporter, pointing to the backend
        exporter = OTLPSpanExporter(
            endpoint=config.endpoint or "http://localhost:4318/v1/traces",
            headers=config.headers,
        )
        
        # wire all the 3 components together: otel provider, batch processor, exporter
        # batch processor is responsible for batching spans and sending them to exporter
        provider.add_span_processor(BatchSpanProcessor(exporter))
        
        # get the actual tracer object we will use to create spans
        self._tracer = provider.get_tracer("ness-ai")
        self._provider = provider
        self._capture_tool_args = config.capture_tool_args

    def start_span(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
        kind: str | None = None,
    ) -> Span:
        
        # get the current span from the contextvar
        # could be any type of span (NoopSpan, InMemorySpan, OTelSpan or None)
        parent = _current_span.get()
        
        # map our kind to otel's enum
        otel_kind = _KIND_MAP.get(kind or "internal", OTelSpanKind.INTERNAL)
        
        # build an OTel Context (parent reference) if parent is an OTelSpan
        ctx = None
        if parent is not None and hasattr(parent, "_otel_span"): # if parent is an OTelSpan
            # any new span should be a child of the parent span
            ctx = trace.set_span_in_context(parent._otel_span)
        
        # ask the OTel tracer to create a real span
        otel_span = self._tracer.start_span(
            name, attributes=attributes or {}, kind=otel_kind, context=ctx
        )
        
        # wrap it in our adapter and return
        return OTelSpan(otel_span, self._capture_tool_args)

    # allow tests / hosts to flush on shutdown
    def shutdown(self) -> None:
        self._provider.shutdown()