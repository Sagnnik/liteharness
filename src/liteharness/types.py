from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Awaitable

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
        "warning",
        "interrupted",
        "plan_turn",
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


QuestionHandler = Callable[[list[dict]], Awaitable[list[dict]]]

# Per-Session runtime hooks (bound on the Session instance, not on the shared
# NessAgentConfig). They let a coding-domain adapter thread its state
# (plan autosave, interruption text)
# back into the domain-agnostic turn loop without the SDK knowing about it.
# Called at the end of a SUCCESSFUL plan-mode turn with the assistant text.
# Used by the adapter to autosave the plan file. 
# Interrupted plan turns do NOT fire this hook — they flow through InterruptHandler / the interrupted SessionEvent so there is exactly one interrupt path.
PlanTurnHandler = Callable[[str], None]

# Called on interruption with the captured partial assistant text; 
# returns the text the adapter wants surfaced on the interrupted SessionEvent
# (None/falsy keeps the original). Default behaviour (when the hook is None)
# is for the SDK to emit the interruption_marker AIMessage itself.
InterruptHandler = Callable[[str], str]

