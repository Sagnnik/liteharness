from __future__ import annotations

from typing import Any

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


# USD per 1M tokens: (input, output, cache_read_ratio, cache_write_ratio)
# Fallback when the provider does not return cost in response metadata.
MODEL_PRICING: dict[str, tuple[float, float, float, float]] = {
    "gpt-4o-mini": (0.15, 0.60, 0.50, 1.0),
    "gpt-4o": (2.50, 10.00, 0.50, 1.0),
    "claude-3.5-sonnet": (3.00, 15.00, 0.10, 1.25),
    "claude-3-haiku": (0.25, 1.25, 0.10, 1.25),
    "deepseek-chat": (0.14, 0.28, 0.10, 1.0),
    "deepseek-v4-flash": (0.09, 0.18, 0.22, 1.0),
    "glm-5.1": (0.98, 3.08, 0.19, 1.0),
    "glm-5.2": (0.95, 3.00, 0.19, 1.0),
    "kimi-k2.6": (0.66, 3.41, 0.50, 1.0),
    "kimi-k2.7-code": (0.74, 3.50, 0.22, 1.0),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_000_000,
    "o3": 200_000,
    "o4-mini": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "deepseek-chat": 128_000,
    "deepseek-v4-flash": 1_000_000,
    "gemini-2.0-flash": 1_000_000,
    "glm-5.1": 203_000,
    "glm-5.2": 1_000_000,
    "kimi-k2.6": 262_000,
    "kimi-k2.7-code": 262_000,
}

VISION_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision",
    "claude-3.5-sonnet",
    "claude-3-opus",
    "claude-3-haiku",
    "gemini-pro-vision",
    "gemini-2.0-flash",
    "glm-5.1",
    "kimi-k2.6",
    "kimi-k2.7-code",
}

# Curated OpenRouter slugs offered by the /config model switcher. Edit freely.
# Substring matching against MODEL_PRICING / MODEL_CONTEXT_WINDOWS keeps cost and
# context-window resolution working for the provider-prefixed slugs below.
AVAILABLE_MODELS: tuple[str, ...] = (
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/gpt-4.1",
    "openai/o4-mini",
    "anthropic/claude-3.5-sonnet",
    "anthropic/claude-3-haiku",
    "google/gemini-2.0-flash",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.7-code",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    model_name: str = Field(default="deepseek-v4-flash", alias="MODEL_NAME")
    reflection_model_name: str = Field(default="deepseek-v4-flash", alias="REFLECTION_MODEL_NAME")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    enable_approval: bool = Field(default=True, alias="ENABLE_APPROVAL")
    auto_save_threads: bool = Field(default=True, alias="AUTO_SAVE_THREADS")
    reflection_token_ratio: float = Field(default=0.4, alias="REFLECTION_TOKEN_RATIO")
    compaction_token_budget: int = Field(default=120_000, alias="COMPACTION_TOKEN_BUDGET")
    compaction_output_reserve_tokens: int = Field(default=8_192, alias="COMPACTION_OUTPUT_RESERVE_TOKENS")
    compaction_input_reserve_tokens: int = Field(default=4_096, alias="COMPACTION_INPUT_RESERVE_TOKENS")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    openrouter_session_id: str | None = Field(default=None, alias="OPENROUTER_SESSION_ID")
    ness_dir: str = Field(default=".ness", alias="NESS_DIR")
    format_on_write: bool = Field(default=True, alias="FORMAT_ON_WRITE")
    exa_api_key: str | None = Field(default=None, alias="EXA_API_KEY")

    @property
    def has_exa(self) -> bool:
        return bool(self.exa_api_key)

    @property
    def supports_vision(self) -> bool:
        model = self.model_name.lower()
        return any(marker in model for marker in VISION_MODELS)


settings = Settings()


def reload_settings() -> None:
    """Re-read environment (including a refreshed .env) into the shared settings.

    Mutates the existing ``settings`` singleton in place so every module that did
    ``from config import settings`` observes the new values.
    """
    load_dotenv(override=True)
    fresh = Settings()
    for field in type(fresh).model_fields:
        setattr(settings, field, getattr(fresh, field))
    from tools.web import reset_provider

    reset_provider()


class CostTracker:
    """Tracks token usage and cache-aware cost for the active process."""

    def __init__(self) -> None:
        self.input_tokens = 0
        self.uncached_input_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
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

        model = model_name or self.model_name or settings.model_name
        self.model_name = model

        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
        cache_read = _detail_value(usage, "input_token_details", "cache_read", "cached_tokens")
        cache_write = _detail_value(usage, "input_token_details", "cache_creation", "cache_write_tokens")
        uncached_input = max(input_tokens - cache_read - cache_write, 0)
        provider_cost = _provider_cost(response_metadata or {})
        estimated_cost = _estimate_cost(model, uncached_input, cache_read, cache_write, output_tokens)
        cost_usd = provider_cost if provider_cost is not None else estimated_cost
        cost_source = "provider" if provider_cost is not None else "estimated"

        self.input_tokens += input_tokens
        self.uncached_input_tokens += uncached_input
        self.cached_input_tokens += cache_read
        self.cache_write_tokens += cache_write
        self.output_tokens += output_tokens
        self.calls += 1
        if cost_usd is not None:
            self.cost_usd += cost_usd

        return {
            "model": model,
            "input_tokens": input_tokens,
            "uncached_input_tokens": uncached_input,
            "cached_input_tokens": cache_read,
            "cache_write_tokens": cache_write,
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

    def report(self, session_id: str | None = None) -> str:
        cost_str = f"${self.cost_usd:.4f}" if self.cost_usd > 0 else "unknown"
        lines: list[str] = []
        if session_id is not None:
            lines.append(f"OpenRouter session: {session_id or 'not set'}")
        lines.extend(
            [
                f"Calls: {self.calls}",
                f"Input tokens: {self.input_tokens:,}",
                f"Uncached input: {self.uncached_input_tokens:,}",
                f"Cached read: {self.cached_input_tokens:,}",
                f"Cache write: {self.cache_write_tokens:,}",
                f"Output tokens: {self.output_tokens:,}",
                f"Total tokens: {self.total_tokens:,}",
                f"Cache hit rate: {self.cache_hit_rate:.1%}",
                f"Cost: {cost_str}",
            ]
        )
        return "\n".join(lines)


def _estimate_cost(
    model_name: str,
    uncached_input_tokens: int,
    cached_input_tokens: int,
    cache_write_tokens: int,
    output_tokens: int,
) -> float | None:
    model = model_name.lower()
    key = next((candidate for candidate in MODEL_PRICING if candidate in model), None)
    if key is None:
        return None
    input_per_m, output_per_m, read_ratio, write_ratio = MODEL_PRICING[key]
    return (
        uncached_input_tokens * input_per_m
        + cached_input_tokens * input_per_m * read_ratio
        + cache_write_tokens * input_per_m * write_ratio
        + output_tokens * output_per_m
    ) / 1_000_000


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


def _value(obj: Any, name: str) -> Any:
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _provider_cost(metadata: dict[str, Any]) -> float | None:
    for key in ("cost", "total_cost", "cache_discount"):
        value = metadata.get(key)
        if value is not None and key != "cache_discount":
            return float(value)
    cost_details = metadata.get("cost_details") or {}
    for key in ("total", "total_cost", "cost"):
        value = cost_details.get(key)
        if value is not None:
            return float(value)
    return None


cost_tracker = CostTracker()
