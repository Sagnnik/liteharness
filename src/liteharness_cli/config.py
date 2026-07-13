from __future__ import annotations

from typing import Any, Literal

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


# USD per 1M tokens: (input, output, cache_read_ratio)
# Fallback when the provider does not return cost in response metadata.
# Order matters for substring matching — list more specific slugs first.
MODEL_PRICING: dict[str, tuple[float, float, float]] = {
    "gpt-5.5": (5.00, 30.00, 0.10),
    "gpt-5.4": (2.50, 15.00, 0.10),
    "gpt-5.2": (1.75, 14.00, 0.10),
    "gpt-5.1": (1.25, 10.00, 0.10),
    "gpt-5": (1.25, 10.00, 0.10),
    "gpt-4o-mini": (0.15, 0.60, 0.50),
    "gpt-4o": (2.50, 10.00, 0.50),
    "gpt-4.1": (2.00, 8.00, 0.25),
    "o4-mini": (1.10, 4.40, 0.25),
    "claude-opus-4.8": (5.00, 25.00, 0.10),
    "claude-opus-4.7": (5.00, 25.00, 0.10),
    "claude-opus-4.6": (5.00, 25.00, 0.10),
    "claude-sonnet-5": (2.00, 10.00, 0.10),
    "claude-sonnet-4.6": (3.00, 15.00, 0.10),
    "claude-sonnet-4.5": (3.00, 15.00, 0.10),
    "claude-sonnet-4": (3.00, 15.00, 0.10),
    "claude-haiku-4.5": (1.00, 5.00, 0.10),
    "claude-3.5-sonnet": (3.00, 15.00, 0.10),
    "claude-3-haiku": (0.25, 1.25, 0.12),
    "gemini-3.1-pro": (2.00, 12.00, 0.10),
    "gemini-2.5-pro": (1.25, 10.00, 0.10),
    "gemini-2.5-flash": (0.30, 2.50, 0.10),
    "gemini-2.0-flash": (0.10, 0.40, 0.10),
    "deepseek-v4-flash": (0.09, 0.18, 0.20),
    "deepseek-chat": (0.20, 0.80, 0.10),
    "kimi-k2.7-code": (0.74, 3.50, 0.20),
    "kimi-k2.6": (0.66, 3.41, 0.21),
    "glm-5.2": (0.69, 2.16, 0.19),
    "glm-5.1": (0.97, 3.04, 0.19),
}

MODEL_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-5.5": 1_050_000,
    "gpt-5.4": 1_050_000,
    "gpt-5.2": 400_000,
    "gpt-5.1": 400_000,
    "gpt-5": 400_000,
    "gpt-4o-mini": 128_000,
    "gpt-4o": 128_000,
    "gpt-4.1": 1_047_576,
    "o4-mini": 200_000,
    "o3": 200_000,
    "claude-opus-4.8": 1_000_000,
    "claude-opus-4.7": 1_000_000,
    "claude-opus-4.6": 1_000_000,
    "claude-sonnet-5": 1_000_000,
    "claude-sonnet-4.6": 1_000_000,
    "claude-sonnet-4.5": 1_000_000,
    "claude-sonnet-4": 1_000_000,
    "claude-haiku-4.5": 200_000,
    "claude-3.5-sonnet": 200_000,
    "claude-3-5-sonnet": 200_000,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-3.1-pro": 1_048_576,
    "gemini-2.5-pro": 1_048_576,
    "gemini-2.5-flash": 1_048_576,
    "gemini-2.0-flash": 1_000_000,
    "deepseek-chat": 131_072,
    "deepseek-v4-flash": 1_048_576,
    "glm-5.2": 1_048_576,
    "glm-5.1": 202_752,
    "kimi-k2.7-code": 262_144,
    "kimi-k2.6": 262_144,
}

VISION_MODELS = {
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-vision",
    "gpt-5",
    "claude-3.5-sonnet",
    "claude-sonnet-4",
    "claude-sonnet-5",
    "claude-opus-4",
    "claude-3-opus",
    "claude-3-haiku",
    "claude-haiku-4.5",
    "gemini-pro-vision",
    "gemini-2.0-flash",
    "gemini-2.5-flash",
    "gemini-3.1-pro",
    "gemini-2.5-pro",
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
    "openai/gpt-5",
    "openai/gpt-5.1",
    "openai/gpt-5.2",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "anthropic/claude-3-haiku",
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-4",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-sonnet-4.6",
    "anthropic/claude-sonnet-5",
    "anthropic/claude-opus-4.6",
    "anthropic/claude-opus-4.7",
    "anthropic/claude-opus-4.8",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "google/gemini-3.1-pro-preview",
    "deepseek/deepseek-chat",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "moonshotai/kimi-k2.7-code",
    "z-ai/glm-5.1",
    "z-ai/glm-5.2",
)

# Per-model reasoning metadata (sync via scripts/fetch_openrouter_models.py).
# Keys are short slugs; matched with `key in model_name.lower()` like pricing.
# Order matters — more specific slugs must appear before shorter prefixes.
# `efforts: ()` = thinking model with no effort selector; `None` = all REASONING_EFFORTS.
MODEL_REASONING: dict[str, dict[str, Any]] = {
    "gpt-5.5": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.4": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.2": {"efforts": ("xhigh", "high", "medium", "low", "none"), "default": "medium", "mandatory": False},
    "gpt-5.1": {"efforts": ("high", "medium", "low", "none"), "default": "none", "mandatory": False},
    "gpt-5": {"efforts": ("high", "medium", "low", "minimal"), "default": "medium", "mandatory": True},
    "o4-mini": {"efforts": ("low", "medium", "high"), "default": None, "mandatory": False},
    "claude-opus-4.8": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-opus-4.7": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-opus-4.6": {"efforts": ("max", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-5": {"efforts": ("max", "xhigh", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-4.6": {"efforts": ("max", "high", "medium", "low"), "default": "medium", "mandatory": False},
    "claude-sonnet-4.5": {"efforts": (), "default": None, "mandatory": False},
    "claude-sonnet-4": {"efforts": (), "default": None, "mandatory": False},
    "claude-haiku-4.5": {"efforts": (), "default": None, "mandatory": False},
    "gemini-3.1-pro": {"efforts": (), "default": None, "mandatory": True},
    "gemini-2.5-pro": {"efforts": (), "default": None, "mandatory": True},
    "gemini-2.5-flash": {"efforts": (), "default": None, "mandatory": False},
    "deepseek-v4-flash": {"efforts": ("xhigh", "high"), "default": "high", "mandatory": False},
    "kimi-k2.7-code": {"efforts": (), "default": None, "mandatory": True},
    "kimi-k2.6": {"efforts": (), "default": None, "mandatory": False},
    "glm-5.2": {"efforts": ("xhigh", "high"), "default": "high", "mandatory": False},
    "glm-5.1": {"efforts": (), "default": None, "mandatory": False},
}

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]
REASONING_EFFORTS: tuple[str, ...] = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="")

    model_name: str = Field(default="deepseek-v4-flash", alias="MODEL_NAME")
    reflection_model_name: str = Field(default="deepseek-v4-flash", alias="REFLECTION_MODEL_NAME")
    reasoning_effort: ReasoningEffort = Field(default="xhigh", alias="REASONING_EFFORT")
    api_max_retries: int = Field(default=3, alias="API_MAX_RETRIES")
    enable_approval: bool = Field(default=True, alias="ENABLE_APPROVAL")
    auto_save_threads: bool = Field(default=True, alias="AUTO_SAVE_THREADS")
    session_end_reflection: bool = Field(default=False, alias="SESSION_END_REFLECTION")
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
    from liteharness.tools.web import reset_provider

    reset_provider()


class CostTracker:
    """Tracks token usage and cache-aware cost for the active process."""

    def __init__(self) -> None:
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

        model = model_name or self.model_name or settings.model_name
        self.model_name = model

        input_tokens = _usage_value(usage, "input_tokens", "prompt_tokens")
        output_tokens = _usage_value(usage, "output_tokens", "completion_tokens")
        total_tokens = _usage_value(usage, "total_tokens") or input_tokens + output_tokens
        cache_read = _detail_value(usage, "input_token_details", "cache_read", "cached_tokens")
        uncached_input = max(input_tokens - cache_read, 0)
        provider_cost = _provider_cost(response_metadata or {})
        estimated_cost = _estimate_cost(model, uncached_input, cache_read, output_tokens)
        cost_usd = provider_cost if provider_cost is not None else estimated_cost
        cost_source = "provider" if provider_cost is not None else "estimated"

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

    def report(self, session_id: str | None = None, *, resume_thread_id: str | None = None) -> str:
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
                f"Output tokens: {self.output_tokens:,}",
                f"Total tokens: {self.total_tokens:,}",
                f"Cache hit rate: {self.cache_hit_rate:.1%}",
                f"Cost: {cost_str}",
            ]
        )
        if resume_thread_id:
            lines.append(f"Resume:  liteharness --resume {resume_thread_id}")
        return "\n".join(lines)


def resolve_model_key(model_name: str, catalog: dict[str, Any]) -> str | None:
    model = model_name.lower()
    return next((candidate for candidate in catalog if candidate in model), None)


def _reasoning_entry(model_name: str) -> dict[str, Any] | None:
    key = resolve_model_key(model_name, MODEL_REASONING)
    if key is None:
        return None
    return MODEL_REASONING[key]


def model_supports_reasoning(model_name: str) -> bool:
    return _reasoning_entry(model_name) is not None


def reasoning_efforts_for_model(model_name: str) -> tuple[str, ...]:
    entry = _reasoning_entry(model_name)
    if entry is None:
        return ()
    efforts = entry.get("efforts")
    supported = REASONING_EFFORTS if efforts is None else tuple(efforts)
    if entry.get("mandatory"):
        supported = tuple(level for level in supported if level != "none")
    return supported


def default_reasoning_effort_for_model(model_name: str) -> str | None:
    efforts = reasoning_efforts_for_model(model_name)
    if not efforts:
        return None
    entry = _reasoning_entry(model_name) or {}
    default = entry.get("default")
    if default and default in efforts:
        return str(default)
    return efforts[0]


def coerce_reasoning_effort(model_name: str, effort: str | None) -> str | None:
    efforts = reasoning_efforts_for_model(model_name)
    if not efforts:
        return None
    if effort and effort in efforts:
        return effort
    return default_reasoning_effort_for_model(model_name)


def _estimate_cost(
    model_name: str,
    uncached_input_tokens: int,
    cached_input_tokens: int,
    output_tokens: int,
) -> float | None:
    key = resolve_model_key(model_name, MODEL_PRICING)
    if key is None:
        return None
    input_per_m, output_per_m, read_ratio = MODEL_PRICING[key]
    return (
        uncached_input_tokens * input_per_m
        + cached_input_tokens * input_per_m * read_ratio
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


def make_sdk_cost_tracker():
    """SDK CostTracker wired with CLI MODEL_PRICING estimates for non-provider costs."""
    from liteharness.tracing.cost import CostTracker as SdkCostTracker

    return SdkCostTracker(pricing=MODEL_PRICING)
