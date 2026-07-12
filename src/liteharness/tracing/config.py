from __future__ import annotations

from dataclasses import dataclass, field

@dataclass
class TracingConfig:
    enabled: bool = False
    service_name: str = "liteharness"
    exporter: str = "otlp"   # "otlp" | "console" | "none"
    endpoint: str | None = None
    headers: dict[str, str] = field(default_factory=dict)
    resource_attrs: dict[str, str] = field(default_factory=dict)
    include_session_attrs: bool = True