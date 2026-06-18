from __future__ import annotations

import asyncio
import json
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Iterator

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from pydantic import BaseModel, Field

from context import build_reflection_prompt, render_todos
from memory import append_session_bullets, load_session_memory

"""
There are 3 Triggers for reflection:
1. Reflection Interval after N turns 
2. Todo Completion
3. Session Exit

There are 2 main jobs for reflection:
1. Distill the recent works into max 2 bullet points -> writes to ./ness/sessions/mem_<thread_id>.md
2. Detect loops -> stores a one shot warning consumed on the next turn
"""

_pending_alerts: dict[str, str] = {}
# tracks locks for each thread_id and prevent overlapping reflection runs due to various triggers
_reflection_locks: dict[str, asyncio.Lock] = {} 
_completed_message_indices: dict[str, int] = {}

# will llm give us both alert message and new bullet points?
class ReflectionStructuredOutput(BaseModel):
    stuck_detected: bool = Field(
        description=(
            "True if the main agent is repeating tool calls, trapped in a loop, "
            "or hitting the same error consecutively."
        )
    )
    alert_message: str = Field(
        default="",
        description=(
            "Intervention message for the main loop when stuck_detected is True. "
            "Otherwise empty."
        ),
    )
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
    stuck_detected: bool = False
    alert_message: str = ""
    error: str = ""


def consume_reflection_alert(thread_id: str) -> str:
    """Pop and return pending alert. Called at the start of each agent_node turn."""
    return _pending_alerts.pop(thread_id, "")


def set_reflection_alert(thread_id: str, message: str) -> None:
    """Store alert if message is not empty"""
    if message.strip():
        _pending_alerts[thread_id] = message.strip()


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
    """Run semantic distillation + loop detection via structured output."""
    # should use a smaller model for reflection
    if model is None:
        return ReflectionResult()

    # get the lock for the thread_id; create it if it doesn't exist
    lock = reflection_lock(thread_id)
    if lock.locked():
        return ReflectionResult()

    # acquire the lock and run the reflection
    async with lock:
        # get the recent messages since the last reflection
        message_list = list(messages)
        since_index = max(0, int(last_reflected_message_index or 0))
        recent_messages = message_list[since_index:]
        loop_hints = _detect_loop_hints(message_list)

        #build the prompt
        prompt = build_reflection_prompt(
            thread_id=thread_id,
            messages=recent_messages,
            user_message_count=user_message_count,
            current_session_bullets=load_session_memory(thread_id),
            todos=todos,
            tool_digest=_build_tool_digest(message_list),
            loop_hints=loop_hints,
        )

        # call the structured output model
        try:
            structured_model = model.with_structured_output(ReflectionStructuredOutput)
            output: ReflectionStructuredOutput = await structured_model.ainvoke(
                [HumanMessage(content=prompt)]
            )
        except Exception as exc:
            return ReflectionResult(error=str(exc))

        # parse the output
        bullets = _normalize_bullets(output.new_bullet_points)
        stuck = output.stuck_detected
        alert = output.alert_message.strip() if stuck else ""
        if stuck and not alert:
            alert = _default_alert_message(loop_hints)

        # append the bullets to the session memory
        memory_updated = append_session_bullets(thread_id, bullets) if bullets else False

        # set the alert if the stuck_detected is True
        if stuck:
            set_reflection_alert(thread_id, alert)

        # record the length of the message list after the successful run
        mark_reflection_complete(thread_id, len(message_list))

        return ReflectionResult(
            memory_updated=memory_updated,
            stuck_detected=stuck,
            alert_message=alert,
        )


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


def _default_alert_message(loop_hints: str) -> str:
    if loop_hints.strip():
        return (
            "SYSTEM WARNING: You appear stuck in a repeated tool/error loop. "
            f"{loop_hints.strip()} Step back, re-read relevant files, and change your approach."
        )
    return (
        "SYSTEM WARNING: Your current strategy appears to be failing or looping. "
        "Step back, re-read relevant files, and change your approach."
    )


def _tool_output_snippet(content: str, limit: int = 240) -> str:
    stripped = content.strip()
    if not stripped:
        return ""
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if len(lines) > 3:
        return lines[-1][:limit]
    return stripped.replace("\n", " ")[:limit]


def _loop_error_key(content: str, limit: int = 160) -> str:
    lines = [line.strip() for line in content.splitlines() if line.strip()] if content else []
    return lines[-1][:limit] if lines else ""


def _iter_tool_rounds(messages: list[BaseMessage]) -> Iterator[tuple[str, str, str]]:
    call_registry: dict[str, tuple[str, str]] = {}
    pending: list[tuple[str, str]] = []

    for message in messages:
        if isinstance(message, AIMessage):
            calls = getattr(message, "tool_calls", None) or []
            for call in calls:
                name = str(call.get("name", ""))
                args_text = _compact_json(call.get("args", {}))
                call_id = call.get("id")
                if call_id:
                    call_registry[str(call_id)] = (name, args_text)
                else:
                    pending.append((name, args_text))
        elif isinstance(message, ToolMessage):
            call_id = getattr(message, "tool_call_id", None)
            if call_id and str(call_id) in call_registry:
                name, args_text = call_registry[str(call_id)]
                yield name, args_text, str(message.content or "")
            elif pending:
                name, args_text = pending.pop(0)
                yield name, args_text, str(message.content or "")


def _build_tool_digest(messages: list[BaseMessage], limit: int = 10) -> str:
    rounds = [
        f"- {name}({args_text}) -> {_tool_output_snippet(content)}"
        for name, args_text, content in _iter_tool_rounds(messages)
    ]
    if not rounds:
        return "(no recent tool activity)"
    return "\n".join(rounds[-limit:])


def _compact_json(value) -> str:
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


async def finalize_session_reflection(
    app,
    thread_id: str,
    model,
) -> ReflectionResult:
    """Run a final synchronous reflection pass before session archive. Called by cli on session exit."""
    try:
        snapshot = await app.aget_state({"configurable": {"thread_id": thread_id}})
    except Exception as exc:
        return ReflectionResult(error=str(exc))

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


def _detect_loop_hints(messages: list[BaseMessage], threshold: int = 3) -> str:
    signatures = []
    for name, args_text, content in _iter_tool_rounds(messages):
        error_key = _loop_error_key(content)
        signatures.append(f"{name}|{args_text}|{error_key}")

    if not signatures:
        return ""

    counts = Counter(signatures)
    hints: list[str] = []
    for signature, count in counts.most_common(3):
        if count < threshold:
            continue
        name, args_text, error_key = signature.split("|", 2)
        hints.append(
            f"Repeated {count}x: tool={name} args={args_text}"
            + (f" error={error_key}" if error_key else "")
        )
    return "\n".join(hints)
