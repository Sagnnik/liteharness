from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Awaitable
from liteharness.compaction import ContextPressure

@dataclass
class UsageEvent:
    model: str
    input_tokens: int
    uncached_input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    cost_usd: float | None

@dataclass(frozen=True)
class SessionEvent:
    kind: Literal[
        "assistant_delta", 
        "assistant_final", 
        "tool_start", 
        "tool_end", 
        "usage",
        "approval_required", 
        "question_required", 
        "compaction", 
        "reflection", 
        "error",
    ]
    data: dict[str, Any]

@dataclass(frozen=True)
class RunResult:
    assistant_message: str
    usage: UsageEvent | None
    todos: list[dict[str, Any]]
    events: list[SessionEvent]


class ApprovalHandler(ABC):
    """Abstract handler for tool-use approval decisions.

    Return one of:
      "yes"     - allow this one call
      "no"      - deny this one call
      "always"  - allow and persist a permanent allow rule
      "session" - allow and persist an allow rule for this session only
      "never"   - deny and persist a permanent deny rule
    """

    @abstractmethod
    async def __call__(self, tool: str, args: dict) -> str: ...


UsageCallback = Callable[[UsageEvent], None]
QuestionHandler = Callable[[list[dict]], Awaitable[list[dict]]]

# For CLI use cases
# Checks file mutations for rollback support
OnFileMutation = Callable[[str, int, str, dict], None]
# Soft pre-act compaction ask: return True to force compact before plan->act execution.
# Accepts a ContextPressure dataclass with token_count, usable_budget, ratio, etc.
PreActCompactHandler = Callable[[ContextPressure], Awaitable[bool]]

