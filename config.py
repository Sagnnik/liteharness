from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from typing import Literal
from pydantic import Field, PastDate
load_dotenv()

class Settings(BaseSettings):
    model_name: str = Field(default="gpt-4o-mini", alias="MODEL_NAME")
    mode: Literal["json", "xml"] = Field(default="json", alias="MODE")
    max_tokens: int = Field(default=120_000, alias="MAX_TOKENS")
    enable_approval: bool = Field(default=True, alias="ENABLE_APPROVAL")
    enable_planning: bool = Field(default=True, alias="ENABLE_PLANNING")
    openai_api_key: str = Field(alias="OPENAI_API_KEY")
    openai_base_url: str | None = Field(default=None, alias="OPENAI_BASE_URL")

    class Config:
        env_prefix = ""

settings = Settings()

# TODO: Implement this! (Need to get the pricing from the model provider)
class CostTracker:
    """Tracks token usage and estimated cost."""

    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.calls = 0

    def add(self, usage):
        if not usage:
            return

        pass

    @property
    def total_cost(self)-> float:
        pass

    def report(self) -> str:
        total = self.prompt_tokens + self.completion_tokens
        pass


cost_tracker = CostTracker()