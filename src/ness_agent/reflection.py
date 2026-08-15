"""
Reflection triggers:
1. Token delta since last reflection exceeds reflection_token_ratio * usable budget
2. Session exit (finalize_session_reflection)

Job: distill recent work into up to 2 bullet points -> .ness/runtime/sessions/mem_<thread_id>.md
Bullets are injected into L3 system-reminder overlay on subsequent turns.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Iterable

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, Field

from ness_agent.tracing.semconv import (
    GEN_AI_COMPLETION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_VALUE,
    KIND_CLIENT,
    REFLECTION,
    THREAD_ID,
)
from ness_agent.tracing.messages import serialize_completion_dict, serialize_messages
from ness_agent.tools.todo import render_todos
from ness_agent.utils import message_to_text

_reflection_locks: dict[str, asyncio.Lock] = {}
_completed_indices: dict[str, int] = {}

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
    bullets: tuple[str, ...] = ()
    message_index: int | None = None

def consume_reflection_message_index(thread_id: str) -> int | None:
    """Pop index written by last successful reflection. Agent stores it in `last_reflection_index`"""
    return _completed_indices.pop(thread_id, None)


def mark_reflection_complete(thread_id: str, message_index: int) -> None:
    """Record len(message_list) after a successful run."""
    _completed_indices[thread_id] = message_index


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


# --- reflection gate ---
def _messages_for_prompt(messages, limit=16) -> str:
    """Join recent messages for the reflection prompt via ``message_to_text``."""
    recent = list(messages)[-limit:]
    return "\n\n".join(
        f"{m.type}: {message_to_text(m)[:1200]}" for m in recent
    )


def _normalize_bullets(bullets: list[str]) -> list[str]:
    """Normalize the bullets to a list of strings."""
    out = []
    for b in bullets:
        t = str(b).strip()
        if t.startswith("- "): 
            t = t[2:].strip()
        if t and t not in out: 
            out.append(t)
        if len(out) >= 2: 
            break
    return out


def _log_reflection_event(
    persistence,
    thread_id: str,
    *,
    prompt: str,
    response: dict[str, Any],
    message_index: int,
    memory_updated: bool,
    error: str,
) -> None:
    if not persistence:
        return
    persistence.append_event(
        thread_id,
        {
            "kind": "reflection",
            "prompt": prompt,
            "response": response,
            "message_index": message_index,
            "memory_updated": memory_updated,
            "error": error,
        },
    )


async def _run_reflection(
    thread_id: str,
    messages: Iterable[BaseMessage],
    model,
    user_message_count: int,
    *,
    last_reflection_index: int = 0,
    todos: str = "",
    memory=None,
    persistence=None,
    aux_prompts=None,
    tracer=None,
    tracing=None,
) -> ReflectionResult:
    """Run semantic distillation while the caller holds the thread lock."""

    if model is None:
        return ReflectionResult()
    msg_list = list(messages)
    since = max(0, int(last_reflection_index or 0))
    recent = msg_list[since:]

    if not recent:
        return ReflectionResult()
    # load_session already returns "- bullet" lines; use as-is.
    bullets_txt = memory.load_session(thread_id) if memory else ""
    tmpl = (aux_prompts.reflection if aux_prompts else None)
    todos_txt = (todos or "").strip() or "No todos"
    
    prompt = tmpl.format(
        thread_id=thread_id,
        user_message_count=user_message_count,
        messages=_messages_for_prompt(recent),
        current_session_bullets=bullets_txt or "(none yet)",
        todos=todos_txt,
    ) if tmpl else _messages_for_prompt(recent)

    try:
        structured_model = model.with_structured_output(ReflectionStructuredOutput)
        capture_msgs = bool(tracing and getattr(tracing, "capture_messages", False))
        if tracer is None:
            out: ReflectionStructuredOutput = await structured_model.ainvoke(
                [HumanMessage(content=prompt)]
            )
        else:
            refl_attrs = {
                THREAD_ID: thread_id,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "chat",
            }
            with tracer.start_span(
                REFLECTION, attributes=refl_attrs, kind=KIND_CLIENT
            ) as span:
                if capture_msgs:
                    span.set_attribute(
                        GEN_AI_PROMPT,
                        serialize_messages([HumanMessage(content=prompt)]),
                    )
                out: ReflectionStructuredOutput = await structured_model.ainvoke(
                    [HumanMessage(content=prompt)]
                )
                if out is not None:
                    span.set_attribute(
                        "reflection.bullets", len(out.new_bullet_points or [])
                    )
                    if capture_msgs:
                        # Reflection uses with_structured_output, so the
                        # completion is a parsed pydantic model (not an
                        # AIMessage) — serialise via the dict helper.
                        span.set_attribute(
                            GEN_AI_COMPLETION,
                            serialize_completion_dict(out),
                        )
    except Exception as exc:
        res = ReflectionResult(error=str(exc))
        _log_reflection_event(
            persistence,
            thread_id,
            prompt=prompt,
            response={"new_bullet_points": []},
            message_index=len(msg_list),
            memory_updated=False,
            error=res.error,
        )
        return res

    bullets = _normalize_bullets(out.new_bullet_points)
    updated = memory.append_session_bullets(thread_id, bullets) if bullets and memory else False
    idx = len(msg_list)

    _log_reflection_event(
        persistence,
        thread_id,
        prompt=prompt,
        response=out.model_dump(),
        message_index=idx,
        memory_updated=updated,
        error="",
    )
    return ReflectionResult(
        memory_updated=updated,
        bullets=tuple(bullets),
        message_index=idx,
    )


async def run_reflection_gate(
    thread_id: str,
    messages: Iterable[BaseMessage],
    model,
    user_message_count: int,
    *,
    last_reflection_index: int = 0,
    todos: str = "",
    memory=None,
    persistence=None,
    aux_prompts=None,
    tracer=None,
    tracing=None,
) -> ReflectionResult:
    """Run automatic reflection, skipping when another pass is active."""
    if model is None:
        return ReflectionResult()
    lock = reflection_lock(thread_id)
    if lock.locked():
        return ReflectionResult()
    async with lock:
        result = await _run_reflection(
            thread_id,
            messages,
            model,
            user_message_count,
            last_reflection_index=last_reflection_index,
            todos=todos,
            memory=memory,
            persistence=persistence,
            aux_prompts=aux_prompts,
            tracer=tracer,
            tracing=tracing,
        )
        if result.message_index is not None:
            mark_reflection_complete(thread_id, result.message_index)
        return result


async def run_session_reflection(
    app,
    thread_id: str,
    model,
    *,
    memory=None,
    persistence=None,
    aux_prompts=None,
    tracer=None,
    tracing=None,
) -> ReflectionResult:
    """Run an explicit reflection pass and persist its graph-state cursor."""
    lock = reflection_lock(thread_id)
    async with lock:
        cfg = {"configurable": {"thread_id": thread_id}}
        try:
            snapshot = await app.aget_state(cfg)
        except Exception as exc:
            result = ReflectionResult(error=str(exc))
            _log_reflection_event(
                persistence,
                thread_id,
                prompt="",
                response={"new_bullet_points": []},
                message_index=0,
                memory_updated=False,
                error=result.error,
            )
            return result

        completed_index = consume_reflection_message_index(thread_id)
        state = dict(snapshot.values or {})
        messages = list(state.get("messages", []))
        state_index = int(state.get("last_reflection_index", 0) or 0)
        since = max(state_index, int(completed_index or 0))
        result = await _run_reflection(
            thread_id,
            messages,
            model,
            sum(1 for message in messages if message.type == "human"),
            last_reflection_index=since,
            todos=render_todos(state.get("todos", [])),
            memory=memory,
            persistence=persistence,
            aux_prompts=aux_prompts,
            tracer=tracer,
            tracing=tracing,
        )

        cursor = result.message_index
        if cursor is None and completed_index is not None:
            cursor = completed_index
        if cursor is not None and cursor != state_index:
            try:
                await app.aupdate_state(cfg, {"last_reflection_index": cursor})
            except Exception as exc:
                mark_reflection_complete(thread_id, cursor)
                return ReflectionResult(
                    memory_updated=result.memory_updated,
                    bullets=result.bullets,
                    message_index=result.message_index,
                    error=f"reflection completed but cursor update failed: {exc}",
                )
        return result

# --- finalize session reflection ---
async def finalize_session_reflection(
    app,
    thread_id: str, model,
    *,
    memory=None,
    persistence=None,
    aux_prompts=None,
    tracer=None,
    tracing=None,
) -> ReflectionResult:
    """Final synchronous reflection pass before session archive (option on)."""
    return await run_session_reflection(
        app,
        thread_id,
        model,
        memory=memory,
        persistence=persistence,
        aux_prompts=aux_prompts,
        tracer=tracer,
        tracing=tracing,
    )
