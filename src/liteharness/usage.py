from __future__ import annotations

from collections.abc import Callable
from typing import Any


def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _usage_value(usage: Any, *names: str) -> int:
    for name in names:
        value = _value(usage, name)
        if value is not None:
            return int(value or 0)
    return 0


def _detail_value(usage: Any, details_key: str, *names: str) -> int:
    details = _value(usage, details_key) or {}
    for name in names:
        value = _value(details, name)
        if value is not None:
            return int(value or 0)
    return 0


def _provider_cost(metadata: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost"):
        value = metadata.get(key)
        if value is not None:
            return float(value)
    cost_details = metadata.get("cost_details") or {}
    for key in ("total", "total_cost", "cost"):
        value = cost_details.get(key)
        if value is not None:
            return float(value)
    return None


class CostTracker:
    """Aggregates token usage. Provider cost preferred; optional estimate_cost for fallback."""

    def __init__(
        self,
        estimate_cost: Callable[[str, int, int, int], float | None] | None = None,
    ) -> None:
        self.estimate_cost = estimate_cost
        self.input_tokens = 0
        self.uncached_input_tokens = 0
        self.cached_input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.cost_usd = 0.0
        self.model_name: str | None = None

    def add(
        self,
        usage: Any,
        model_name: str | None = None,
        response_metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if not usage:
            return None

        model = model_name or self.model_name or ""
        self.model_name = model

        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
        cache_read = _detail_value(usage, "input_token_details", "cache_read", "cached_tokens")
        uncached_input = max(input_tokens - cache_read, 0)
        provider_cost = _provider_cost(response_metadata or {})
        estimated_cost = (
            self.estimate_cost(model, uncached_input, cache_read, output_tokens)
            if self.estimate_cost is not None
            else None
        )
        cost_usd = provider_cost if provider_cost is not None else estimated_cost
        if provider_cost is not None:
            cost_source = "provider"
        elif estimated_cost is not None:
            cost_source = "estimated"
        else:
            cost_source = None

        self.input_tokens += input_tokens
        self.uncached_input_tokens += uncached_input
        self.cached_input_tokens += cache_read
        self.output_tokens += output_tokens
        self.calls += 1
        if cost_usd is not None:
            self.cost_usd += cost_usd

        return {
            "model": model,
            "input_tokens": input_tokens,
            "uncached_input_tokens": uncached_input,
            "cached_input_tokens": cache_read,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "cost_usd": cost_usd,
            "cost_source": cost_source,
            "cache_hit_rate": cache_read / input_tokens if input_tokens else 0.0,
        }

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cache_hit_rate(self) -> float:
        if self.input_tokens <= 0:
            return 0.0
        return self.cached_input_tokens / self.input_tokens

    @property
    def total_cost_usd(self) -> float | None:
        return self.cost_usd if self.cost_usd > 0 else None
