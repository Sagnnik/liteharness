from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Literal
from pydantic import Field
load_dotenv()

# USD per 1M tokens (input, output)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3.5-sonnet": (3.00, 15.00),
    "claude-3-haiku": (0.25, 1.25),
    "deepseek-chat": (0.14, 0.28),
}
VISION_MODELS = {
    "gpt-4o", "gpt-4o-mini", "gpt-4-vision-preview",
    "claude-3.5-sonnet", "claude-3-opus", "claude-3-haiku",
    "gemini-pro-vision", "gemini-2.0-flash",
}

class Settings(BaseSettings):
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    mode: Literal["json", "xml"] = Field(default="json", alias="MODE")
    max_tokens: int = Field(default=120_000, alias="MAX_TOKENS")
    enable_approval: bool = Field(default=True, alias="ENABLE_APPROVAL")
    enable_planning: bool = Field(default=True, alias="ENABLE_PLANNING")
    auto_save_threads: bool = Field(default=True, alias="AUTO_SAVE_THREADS")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")
    ness_dir: str = Field(default=".ness", alias="NESS_DIR")
    
    class Config:
        env_prefix = ""
    
    @property
    def supports_vision(self) -> bool:
        return any(v in self.model_name.lower() for v in VISION_MODELS)


settings = Settings()

class CostTracker:
    """Tracks token usage and estimated cost."""
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, usage):
        if not usage:
            return

        inp = getattr(usage, "input_tokens", None) or usage.get("input_tokens", 0)
        out = getattr(usage, "output_tokens", None) or usage.get("output_tokens", 0)
        self.prompt_tokens += inp
        self.completion_tokens += out
        self.calls += 1

    @property
    def total_cost(self)-> float:
        key = next((k for k in MODEL_PRICING if k in settings.model_name.lower()), None)
        if not key:
            return 0.0
        inp_price, out_price = MODEL_PRICING[key]
        return (self.prompt_tokens * inp_price + self.completion_tokens * out_price) / 1_000_000

    def report(self) -> str:
        total = self.prompt_tokens + self.completion_tokens
        cost = self.total_cost
        cost_str = f"${cost:.2f}" if cost > 0 else "N/A"
        return (
            f"Calls: {self.calls}\n"
            f"Input Tokens: {self.prompt_tokens:,}\n"
            f"Output Tokens: {self.completion_tokens:,}\n"
            f"Total Tokens: {total:,}\n"
            f"Est. Cost: {cost_str}"
        )


cost_tracker = CostTracker()