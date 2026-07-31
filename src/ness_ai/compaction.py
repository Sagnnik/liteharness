from __future__ import annotations

import re
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from ness_ai.tracing.semconv import (
    COMPACTION_SUMMARIZE,
    GEN_AI_COMPLETION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_VALUE,
    THREAD_ID,
    KIND_CLIENT,
)
from ness_ai.tracing.messages import serialize_completion, serialize_messages

# --- tool output truncation ---
_SMALL_CHARS = 900
_SMALL_LINES = 20
_MAX_BODY_CHARS = 2400
_HEAD_LINES = 8
_TAIL_LINES = 4
_MAX_ERROR_LINES = 12

# --- summarization ---
_SUMMARY_MESSAGE_LIMIT = 40
_FALLBACK_SUMMARY_CHARS = 8000

# --- pressure thresholds ---
# Summary compaction triggers at 80% (not at the ceiling): the summarizing model
# is already degraded by context rot past this point, so compact before it worsens.
COMPACTION_TOOL_RATIO = 0.70
COMPACTION_SUMMARY_RATIO = 0.80
COMPACTION_HARD_RATIO = 0.92

# --- keep-count ---
_MIN_KEEP_RECENT = 2
_MAX_KEEP_RECENT = 10

# --- manual force floor ---
_FORCE_SUMMARY_MIN_MESSAGES = 10
_FORCE_SUMMARY_KEEP = 10

_ERROR_RE = re.compile(
    r"(error|failed|exception|traceback|denied|fatal|panic|cannot|"
    r"command not found|no such file|permission denied|exit code|exit status)",
    re.IGNORECASE,
)

CompactionAction = Literal["none", "tool_outputs", "summary"]

_ACTION_RANK: dict[CompactionAction, int] = {
    "none": 0,
    "tool_outputs": 1,
    "summary": 2,
}


@dataclass(frozen=True)
class CompactionResult:
    messages: list[BaseMessage]
    compacted: bool
    token_count: int
    action: CompactionAction = "none"
    kept_recent: int = 0
    summary: str = ""
    pressure_ratio: float = 0.0
    usable_budget: int = 0


@dataclass(frozen=True)
class ContextPressure:
    token_count: int
    usable_budget: int
    ratio: float
    action: CompactionAction
    keep_recent: int
    hard_threshold_reached: bool


def _content_text(content) -> str:
    if isinstance(content, list):
        parts: list[str] = []
        for i in content:
            if isinstance(i, dict):
                parts.append(str(i.get("text", i)))
            else:
                parts.append(str(i))
        return " ".join(parts)
    return str(content)


def _estimate_tokens_symbol(messages: list[BaseMessage]) -> int:
    # cheap token count estimate by counting symbols
    text = "\n\n".join(f"{m.type}: {_content_text(m.content)}" for m in messages)
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return max(1, len(text) // 3 + symbol_count // 2 + 6 * len(messages))


def resolve_token_count(messages: list[BaseMessage], *, known_input_tokens: int | None) -> int:
    # known input tokens is the exact value from the api metadata
    if known_input_tokens and known_input_tokens > 0:
        return known_input_tokens
    return _estimate_tokens_symbol(messages)


def resolve_usable_context_budget(
    model_name: str,
    options=None,
) -> int:
    # if context window is provided, use it else fall back to compaction_token_budget
    _ = model_name  # reserved for future model-aware defaults
    if options is None:
        return 120_000
    window = options.context_window or 0
    if not window:
        return options.compaction_token_budget

    # calculate usable budget = window - reserve
    reserve = options.compaction_output_reserve + options.compaction_input_reserve
    usable = window - reserve
    # for smaller models we return the exact window size
    if usable <= 0:
        return window
    return usable


def compaction_action_for_ratio(ratio: float) -> tuple[CompactionAction, int]:
    """
    1. if ratio < 0.70, we will not compact the conversation
    2. if ratio < 0.80, we will compact the conversation to tool outputs
    3. if ratio >= 0.80, we will compact the conversation to summary
    return the action and the number of recent messages to keep
    """
    if ratio < COMPACTION_TOOL_RATIO: 
        return "none", 0

    if ratio < COMPACTION_SUMMARY_RATIO: 
        return "tool_outputs", 0
    # at/above the summary threshold, summarize with an LLM. Keep _MAX_KEEP_RECENT
    # recent messages at the threshold, decaying to _MIN_KEEP_RECENT as pressure rises.
    span = max(1.0 - COMPACTION_SUMMARY_RATIO, 1e-6)
    keep = max(
        _MIN_KEEP_RECENT,
        min(_MAX_KEEP_RECENT, int(_MAX_KEEP_RECENT * (1.0 - ratio) / span)),
    )
    return "summary", keep


def calculate_context_pressure(
    messages: list[BaseMessage], 
    *, 
    known_input_tokens=None, 
    model_name=None,
    options=None,
    max_tokens: int | None = None,
) -> ContextPressure:
    """
    How full is the context window and what's the compaction action?
    budget: how many tokens can the conversation use
    token count: how many tokens we think are in the conversation
    ratio: token count / budget
    action: 'none', 'tool_outputs', 'summary' + how many recent messages to keep
    """
    if max_tokens and max_tokens > 0:
        budget = int(max_tokens)
    else:
        budget = resolve_usable_context_budget(model_name or "", options)
    
    tc = resolve_token_count(messages, known_input_tokens=known_input_tokens)
    ratio = tc / budget if budget > 0 else 1.0
    action, keep = compaction_action_for_ratio(ratio)
    return ContextPressure(
        token_count=tc,
        usable_budget=budget,
        ratio=ratio,
        action=action,
        keep_recent=keep,
        hard_threshold_reached=ratio >= COMPACTION_HARD_RATIO,
    )

# --- tool output compaction ---
def _summarize_tool_output(content: str, tool_name: str = "") -> str:
    """
    Summarize the content of a tool output message.
    If the content is small (<= 900 chars and <= 20 lines), return the content.
    Otherwise, summarize the content; get the head, tail and error lines (by regex search)
    and return the summarized content.
    """
    # if the content is small (<= 900 chars and <= 20 lines), return the content
    lines = content.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if len(content) <= _SMALL_CHARS and len(nonempty) <= _SMALL_LINES: 
        return content

    # otherwise, summarize the content; get the head, tail and error lines (by regex search)   
    head = nonempty[:_HEAD_LINES]
    tail = nonempty[-_TAIL_LINES:] if len(nonempty) > _TAIL_LINES else []
    errors = [ln for ln in nonempty if _ERROR_RE.search(ln)][:_MAX_ERROR_LINES]
    seen = set() 
    kept = []
    
    for block in (head, errors, tail):
        for l in block:
            if l in seen: continue
            seen.add(l)
            kept.append(l)
    
    if len(kept) >= len(nonempty): 
        return content

    # join the kept lines into a single string or body
    # if the body is too long, truncate it with ...[truncated]
    body = "\n".join(kept)
    if len(body) > _MAX_BODY_CHARS: 
        body = body[:_MAX_BODY_CHARS] + "\n...[truncated]"
    
    return (
        f"[compacted {tool_name or 'tool'} output] "
        f"{len(nonempty)} lines, {len(content)} chars\n"
        f"{body}"
    )


def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    # find ToolMessage and summarize the content
    if not isinstance(message, ToolMessage):
        return message

    content = _summarize_tool_output(str(message.content), message.name or "")
    return message.model_copy(update={"content": content})


# --- force floor & labels ---
def apply_force_floor(
    action: CompactionAction,
    keep_recent: int,
    message_count: int,
) -> tuple[CompactionAction, int]:
    # this is used for force compacting the conversation
    # if >10 messages, we will compact to summary
    if message_count > _FORCE_SUMMARY_MIN_MESSAGES:
        left_rank = _ACTION_RANK[action]
        right_rank = _ACTION_RANK["summary"]
        if right_rank > left_rank:
            # if summary is more aggressive, we will compact to summary
            return "summary", _FORCE_SUMMARY_KEEP
        return action, keep_recent
    # if action is none, we will compact to tool outputs
    if action == "none":
        return "tool_outputs", 0
    return action, keep_recent


def compaction_label(action: CompactionAction, kept_recent: int = 0) -> str:
    if action == "none":
        return "none"
    if action == "tool_outputs":
        return "tool outputs only"
    return f"summary, kept last {kept_recent} messages"


async def progressive_compact(
    messages: list[BaseMessage],
    *,
    known_input_tokens:int | None,
    summary_model=None,
    force: bool = False,
    model_name:str | None = None,
    thread_id: str | None = None,
    options=None,
    persistence=None,
    cost_tracker=None,
    tracer=None,
    tracing=None,
    compaction_prompt=None,
    max_tokens=None
) -> CompactionResult:
    """Compact the messages progressively.
    
    Args:
        messages: The messages to compact.
        known_input_tokens: The known input tokens.
        summary_model: The model to use for summarization.
        force: Whether to force compacting the conversation.
        model_name: The name of the model.
        thread_id: The thread ID.
        options: The options for the conversation.
        persistence: The persistence to use for compaction.
        cost_tracker: The cost tracker to use for compaction.
        tracer: The tracer to use for compaction.
        compaction_prompt: The compaction prompt to use for compaction.
        max_tokens: The maximum tokens to use for compaction.

    Returns:
        CompactionResult -> The compaction result.
    """
    # calculate the context pressure
    pressure = calculate_context_pressure(
        messages, 
        known_input_tokens=known_input_tokens,
        model_name=model_name, 
        options=options, 
        max_tokens=max_tokens
    )

    # get the compaction action and how many recent messages to keep
    action = pressure.action 
    keep = pressure.keep_recent
    system = [m for m in messages if m.type == "system"]
    rest = [m for m in messages if m.type != "system"]
    # if force is True, apply the force floor
    if force: 
        action, keep = apply_force_floor(action, keep, len(rest))
    
    # if action is none, return the original messages
    if action == "none":
        return CompactionResult(
            messages=messages, 
            compacted=False, 
            token_count=pressure.token_count, 
            action=action, 
            kept_recent=keep,
            pressure_ratio=pressure.ratio, 
            usable_budget=pressure.usable_budget,
        )

    # compact the tool messages
    rest = [_compact_tool_message(m) for m in rest]

    if action == "tool_outputs":
        return CompactionResult(
            messages=system + rest, 
            compacted=True, 
            token_count=pressure.token_count, 
            action=action, 
            pressure_ratio=pressure.ratio, 
            usable_budget=pressure.usable_budget,
        )
    # get the min no. of messages to keep and summarize the older messages
    k = min(len(rest), keep)
    older = rest[: len(rest) - k]
    summary = await summarize_history(
        older,
        summary_model,
        thread_id=thread_id,
        action=action,
        kept_recent=k,
        task_prompt=compaction_prompt,
        persistence=persistence,
        cost_tracker=cost_tracker,
        model_name=model_name,
        tracer=tracer,
        tracing=tracing,
    )
    # add the system messages and the summary and the recent messages
    compacted = list(system) + [SystemMessage(content="COMPACTED HISTORY\n" + summary)] + (rest[-k:] if k else [])

    return CompactionResult(
        messages=compacted,
        compacted=True,
        token_count=pressure.token_count,
        action=action,
        kept_recent=keep,
        summary=summary,
        pressure_ratio=pressure.ratio,
        usable_budget=pressure.usable_budget,
    )


def _fallback_summary(serialized: str) -> str:
    excerpt = serialized[:_FALLBACK_SUMMARY_CHARS]
    if len(serialized) > _FALLBACK_SUMMARY_CHARS: excerpt += "\n...[truncated]"
    return "[Compaction summary unavailable]\n" + excerpt


def _persist_compaction_llm(
    persistence,
    thread_id: str | None,
    *,
    prompt: str,
    response: str,
    action: CompactionAction,
    kept_recent: int,
) -> None:
    if not persistence or not thread_id:
        return
    persistence.append_event(
        thread_id,
        {
            "kind": "compaction_llm",
            "prompt": prompt,
            "response": response,
            "action": action,
            "kept_recent": kept_recent,
        },
    )


async def summarize_history(
    messages: Iterable[BaseMessage],
    model,
    *,
    thread_id: str | None = None,
    action: CompactionAction = "summary",
    kept_recent: int = 0,
    task_prompt=None,
    persistence=None,
    cost_tracker=None,
    model_name=None,
    tracer=None,
    tracing=None,
) -> str:
    """Summarize the history of the conversation.

    Args:
        messages: The messages to summarize.
        model: The model to use for summarization.
        thread_id: The thread ID.
        action: The action to take.
        kept_recent: The number of recent messages to keep.
        task_prompt: The task prompt to use for summarization.
        persistence: The persistence to use for summarization.
        cost_tracker: The cost tracker to use for summarization.
        model_name: The name of the model.
        tracer: Optional tracer backend for the summarisation span.

    Returns:
        str -> The summary of the conversation.
    """

    # serialize the messages for compaction
    recent = list(messages)[-_SUMMARY_MESSAGE_LIMIT:]
    serialized = "\n\n".join(f"{m.type}: {_content_text(m.content)}" for m in recent)
    if not serialized.strip():
        return ""

    if model is None:
        # if no model, use the fallback summary (no durable llm row)
        return _fallback_summary(serialized)

    # build the prompt
    prompt = task_prompt.format(messages=serialized) if task_prompt else serialized

    span_attrs: dict[str, Any] = {
        "compaction.action": action,
        "compaction.kept_recent": kept_recent,
        GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
        GEN_AI_OPERATION_NAME: "chat",
    }
    if thread_id:
        span_attrs[THREAD_ID] = thread_id
    if model_name:
        span_attrs["gen_ai.request.model"] = model_name

    capture_msgs = bool(tracing and getattr(tracing, "capture_messages", False))
    span_cm = (
        tracer.start_span(COMPACTION_SUMMARIZE, attributes=span_attrs, kind=KIND_CLIENT)
        if tracer is not None
        else nullcontext()
    )

    summary = ""
    resp = None
    with span_cm as span:
        if capture_msgs and span is not None:
            # The compaction prompt is a single templated user turn, not the raw conversation
            span.set_attribute(
                GEN_AI_PROMPT,
                serialize_messages([HumanMessage(content=prompt)]),
            )
        try:
            resp = await model.ainvoke([HumanMessage(content=prompt)])
        except Exception as exc:
            if span is not None:
                span.record_exception(exc)
                span.set_status("ERROR", str(exc))
            summary = _fallback_summary(serialized)
        else:
            if capture_msgs and span is not None:
                span.set_attribute(GEN_AI_COMPLETION, serialize_completion(resp))
            summary = str(resp.content).strip()
            if resp.usage_metadata and cost_tracker:
                cost_tracker.add(
                    resp.usage_metadata, model_name, resp.response_metadata or {}
                )

    _persist_compaction_llm(
        persistence,
        thread_id,
        prompt=prompt,
        response=summary,
        action=action,
        kept_recent=kept_recent,
    )
    return summary


def compaction_overlay_note(
    result: CompactionResult,
    *,
    budget: int | None = None,
    options=None,
    had_stored_compaction: bool = False,
    warn_ratio: float = 0.7,
    model_name: str | None = None,
) -> str:
    """Build a short L3 note about compaction state and approaching pressure.

    Args:
        result: The compaction result.
        budget: The budget for the conversation.
        options: The options for the conversation.
        had_stored_compaction: Whether there was a stored compaction.
        warn_ratio: The warning ratio.
        model_name: The name of the model.

    Returns:
        A short L3 note about compaction state and approaching pressure.
    """

    parts = []
    if budget is None:
        if options is not None:
            budget = resolve_usable_context_budget(model_name or "", options)
        else:
            budget = 120_000
    ratio = result.pressure_ratio

    if ratio <= 0 and budget > 0:
        ratio = result.token_count / budget

    if result.compacted:
        detail = compaction_label(result.action, result.kept_recent)
        parts.append(
            f"Conversation compacted this turn ({detail}). "
            "Re-read files if you need earlier details."
        )

    elif had_stored_compaction:
        parts.append(
            "Conversation includes compacted history from a prior turn. "
            "Earlier tool output may be summarized."
        )

    if not result.compacted and ratio >= warn_ratio:
        pct = int(100 * ratio)
        warning = (
            "Summary compaction may begin soon."
            if ratio >= (COMPACTION_SUMMARY_RATIO - 0.05)
            else "Tool-output compaction may begin soon."
        )
        parts.append(
            f"Context ~{result.token_count:,} tokens ({pct}% of usable budget). "
            f"{warning}"
        )
    return " ".join(parts)