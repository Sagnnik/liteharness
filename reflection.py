from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from memory import append_memory, write_task_state
from prompt import build_reflection_prompt


class ReflectionPayload(BaseModel):
    durable_learnings: list[str] = Field(default_factory=list)
    volatile_task_state: list[str] = Field(default_factory=list)


@dataclass(frozen=True)
class ReflectionResult:
    durable_changed: bool = False
    state_changed: bool = False
    error: str = ""


async def run_reflection_gate(thread_id: str, messages: Iterable[BaseMessage], model, user_message_count: int) -> ReflectionResult:
    """Classify durable memory vs volatile task state.
    Durable memory --> .ness/NESS.md (appended)
    Volatile task state --> .ness/STATE.md (replaced at each reflection)
    """

    if model is None:
        return ReflectionResult()
    prompt = build_reflection_prompt(thread_id, messages, user_message_count)
    try:
        structured = model.with_structured_output(ReflectionPayload)
        payload: ReflectionPayload = await structured.ainvoke([HumanMessage(content=prompt)])
    except Exception as exc:
        return ReflectionResult(error=str(exc))

    durable = _clean_items(payload.durable_learnings)
    volatile = _clean_items(payload.volatile_task_state)

    durable_changed = False
    state_changed = False
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    if durable:
        append_memory(f"\n## Reflection {stamp}\n" + "\n".join(f"- {item}" for item in durable))
        durable_changed = True
    if volatile:
        write_task_state(
            "# Current Task State\n\n"
            f"Updated: {stamp}\n"
            f"Thread: {thread_id}\n\n"
            + "\n".join(f"- {item}" for item in volatile)
        )
        state_changed = True
    return ReflectionResult(durable_changed=durable_changed, state_changed=state_changed)


def _clean_items(items: list[str]) -> list[str]:
    skip = {"none", "n/a", "no"}
    return [text for text in (item.strip() for item in items) if text and text.lower() not in skip]
