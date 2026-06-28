from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from context import build_reflection_prompt, render_todos
from memory import append_session_bullets, load_session_memory
from session import append_event

"""
Reflection triggers:
1. Token delta since last reflection exceeds reflection_token_ratio * usable budget
2. Session exit (finalize_session_reflection)

Job: distill recent work into up to 2 bullet points -> ./ness/sessions/mem_<thread_id>.md
Bullets are injected into L3 system-reminder overlay on subsequent turns.
"""

_reflection_locks: dict[str, asyncio.Lock] = {}
_completed_message_indices: dict[str, int] = {}


class ReflectionStructuredOutput(BaseModel):
    new_bullet_points: list[str] = Field(
        default_factory=list,
        description=(
            "Up to 2 concise bullets: features added, tasks completed, errors hit, "
            "or conventions discovered."
        ),
    )


@dataclass(frozen=True)
class ReflectionResult:
    memory_updated: bool = False
    error: str = ""


def consume_reflection_message_index(thread_id: str) -> int | None:
    """Pop index written by last successful reflection. Agent stores it in `last_reflected_message_index`"""
    return _completed_message_indices.pop(thread_id, None)


def mark_reflection_complete(thread_id: str, message_index: int) -> None:
    """Record len(message_list) after a successful run."""
    _completed_message_indices[thread_id] = message_index


def reflection_lock(thread_id: str) -> asyncio.Lock:
    """Get or create a lock for the thread_id and store it in _reflection_locks"""
    lock = _reflection_locks.get(thread_id)
    if lock is None:
        lock = asyncio.Lock()
        _reflection_locks[thread_id] = lock
    return lock


def is_reflection_running(thread_id: str) -> bool:
    """Returns True if lock exists and is held. Useful to skip scheduling while a run is active."""
    lock = _reflection_locks.get(thread_id)
    return bool(lock and lock.locked())


async def run_reflection_gate(
    thread_id: str,
    messages: Iterable[BaseMessage],
    model,
    user_message_count: int,
    *,
    last_reflected_message_index: int = 0,
    todos: str = "",
) -> ReflectionResult:
    """Run semantic distillation via structured output."""
    if model is None:
        return ReflectionResult()

    lock = reflection_lock(thread_id)
    if lock.locked():
        return ReflectionResult()

    async with lock:
        message_list = list(messages)
        since_index = max(0, int(last_reflected_message_index or 0))
        recent_messages = message_list[since_index:]

        if not recent_messages:
            return ReflectionResult()

        prompt = build_reflection_prompt(
            thread_id=thread_id,
            messages=recent_messages,
            user_message_count=user_message_count,
            current_session_bullets=load_session_memory(thread_id),
            todos=todos,
        )

        try:
            structured_model = model.with_structured_output(ReflectionStructuredOutput)
            output: ReflectionStructuredOutput = await structured_model.ainvoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as exc:
            result = ReflectionResult(error=str(exc))
            _log_reflection_event(
                thread_id,
                prompt=prompt,
                response={"new_bullet_points": []},
                usage=None,
                message_index=len(message_list),
                memory_updated=False,
                error=result.error,
            )
            return result

        bullets = _normalize_bullets(output.new_bullet_points)
        memory_updated = append_session_bullets(thread_id, bullets) if bullets else False
        message_index = len(message_list)
        mark_reflection_complete(thread_id, message_index)

        _log_reflection_event(
            thread_id,
            prompt=prompt,
            response=output.model_dump(),
            usage=None,
            message_index=message_index,
            memory_updated=memory_updated,
            error="",
        )
        return ReflectionResult(memory_updated=memory_updated)


def _log_reflection_event(
    thread_id: str,
    *,
    prompt: str,
    response: dict[str, Any],
    usage: dict[str, Any] | None,
    message_index: int,
    memory_updated: bool,
    error: str,
) -> None:
    event: dict[str, Any] = {
        "kind": "reflection",
        "prompt": prompt,
        "response": response,
        "message_index": message_index,
        "memory_updated": memory_updated,
        "error": error,
    }
    if usage:
        event.update(usage)
    append_event(thread_id, event)


def _normalize_bullets(bullets: list[str]) -> list[str]:
    cleaned: list[str] = []
    for item in bullets:
        text = str(item).strip()
        if text.startswith("- "):
            text = text[2:].strip()
        if text and text not in cleaned:
            cleaned.append(text)
        if len(cleaned) >= 2:
            break
    return cleaned


async def finalize_session_reflection(
    app,
    thread_id: str,
    model,
) -> ReflectionResult:
    """Run a final synchronous reflection pass before session archive. Called by cli on session exit."""
    try:
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        result = ReflectionResult(error=str(exc))
        _log_reflection_event(
            thread_id,
            prompt="",
            response={"new_bullet_points": []},
            usage=None,
            message_index=0,
            memory_updated=False,
            error=result.error,
        )
        return result

    state = dict(snapshot.values or {})
    messages = list(state.get("messages", []))
    if not messages:
        return ReflectionResult()

    user_count = sum(1 for message in messages if message.type == "human")

    return await run_reflection_gate(
        thread_id,
        messages,
        model,
        user_count,
        last_reflected_message_index=int(state.get("last_reflected_message_index", 0) or 0),
        todos=render_todos(state.get("todos", [])),
    )
