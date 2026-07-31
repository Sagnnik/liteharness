"""Tracing exporter backends.

Modules here are imported lazily by :func:`ness_agent.tracing.tracer.build_tracer`
so the optional ``opentelemetry`` dependency is only required when ``exporter
== "otlp"`` is actually selected.
"""