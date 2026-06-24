from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import cost_tracker, settings, MODEL_CONTEXT_WINDOWS
from context import _content_text, build_compaction_prompt

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
COMPACTION_TOOL_RATIO = 0.70
COMPACTION_SUMMARY_RATIO = 0.85
COMPACTION_HARD_RATIO = 0.92
PLAN_COMPACTION_CHECKPOINT_RATIO = 0.75

# --- keep-count ---
_MIN_KEEP_RECENT = 4
_MAX_KEEP_RECENT = 10

# --- manual force floor ---
_FORCE_SUMMARY_MIN_MESSAGES = 10
_FORCE_SUMMARY_KEEP = 10

CompactionAction = Literal["none", "tool_outputs", "summary"]

_ACTION_RANK: dict[CompactionAction, int] = {
    "none": 0,
    "tool_outputs": 1,
    "summary": 2,
}

_ERROR_RE = re.compile(
    r"(error|failed|exception|traceback|denied|fatal|panic|cannot|"
    r"command not found|no such file|permission denied|exit code|exit status)",
    re.IGNORECASE,
)


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


# --- pressure & budget ---
def _model_context_window(model_name: str) -> int | None:
    model = model_name.lower()
    return next((window for key, window in MODEL_CONTEXT_WINDOWS.items() if key in model), None)


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
    max_tokens: int | None = None,
    *,
    model_name: str | None = None,
) -> int:
    # max tokens are used in tests only
    if max_tokens and max_tokens > 0:
        return int(max_tokens)
    # if window is there then calculate usable budget = window - reserve (set in settings)
    # else use the default budget
    window = _model_context_window(model_name or settings.model_name)
    if not window:
        return int(settings.compaction_token_budget)

    reserve = int(settings.compaction_output_reserve_tokens) + int(settings.compaction_input_reserve_tokens)
    usable = int(window) - reserve
    # for smaller models with we return the exact window size
    if usable <= 0:
        return int(window)
    return usable


def compaction_action_for_ratio(ratio: float) -> tuple[CompactionAction, int]:
    # if ration < 0.70, we will not compact the conversation
    if ratio < COMPACTION_TOOL_RATIO:
        return "none", 0
    # if ration < 0.85, we will compact the conversation to tool outputs only
    if ratio < COMPACTION_SUMMARY_RATIO:
        return "tool_outputs", 0
    # if ration >= 0.85, we will compact the conversation to a LLM generated summary
    # we will keep the last 10 -> 4 messages based on the ratio rest would be summarized
    keep = max(
        _MIN_KEEP_RECENT,
        int(_MAX_KEEP_RECENT * (1.0 - ratio) / 0.15),
    )
    return "summary", keep


def calculate_context_pressure(
    messages: list[BaseMessage],
    max_tokens: int | None = None,
    *,
    known_input_tokens: int | None = None,
    model_name: str | None = None,
) -> ContextPressure:
    """
    How full is the context window and what's the compaction action?
    budget: how many tokens can the conversation use
    token count: how many tokens we think are in the conversation
    ratio: token count / budget
    action: 'none', 'tool_outputs', 'summary' + how many recent messages to keep
    example: 
    context window: 120,000 tokens
    we reserve around 12000 (set in settings) tokens per turn for output and input
    so we can use 108000 tokens for the conversation
    lets say we already used 60000 tokens
    ratio = 60000 / 108000 = 0.56
    since 0.56 < 0.70, we will not compact the conversation
    """
    budget = resolve_usable_context_budget(max_tokens, model_name=model_name)
    token_count = resolve_token_count(messages, known_input_tokens=known_input_tokens)
    ratio = token_count / budget if budget > 0 else 1.0
    action, keep_recent = compaction_action_for_ratio(ratio)
    return ContextPressure(
        token_count=token_count,
        usable_budget=budget,
        ratio=ratio,
        action=action,
        keep_recent=keep_recent,
        hard_threshold_reached=ratio >= COMPACTION_HARD_RATIO,
    )

# --- LLM summarization ---
def _serialize_for_compaction(messages: Iterable[BaseMessage]) -> str:
    recent = list(messages)[-_SUMMARY_MESSAGE_LIMIT:]
    return "\n\n".join(
        f"{message.type}: {_content_text(message.content)}" for message in recent
    )

async def summarize_history(
    messages: Iterable[BaseMessage],
    model,
    *,
    thread_id: str | None = None,
    action: CompactionAction = "summary",
    kept_recent: int = 0,
) -> str:
    # serialize the messages for compaction
    serialized = _serialize_for_compaction(messages)
    if not serialized.strip():
        return ""
    if model is None:
        # if no model, use the fallback summary
        return _fallback_summary(serialized)
    prompt = build_compaction_prompt(serialized)
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception:
        summary = _fallback_summary(serialized)
        if thread_id:
            _log_compaction_event(
                thread_id,
                prompt=prompt,
                response=summary,
                usage=None,
                action=action,
                kept_recent=kept_recent,
            )
        return summary

    usage = None
    if response.usage_metadata:
        usage = cost_tracker.add(
            response.usage_metadata,
            _model_name(model),
            response.response_metadata or {},
        )
    summary = str(response.content).strip()
    if thread_id:
        _log_compaction_event(
            thread_id,
            prompt=prompt,
            response=summary,
            usage=usage,
            action=action,
            kept_recent=kept_recent,
        )
    return summary


def _log_compaction_event(
    thread_id: str,
    *,
    prompt: str,
    response: str,
    usage: dict[str, Any] | None,
    action: CompactionAction,
    kept_recent: int,
) -> None:
    from session import append_event

    event: dict[str, Any] = {
        "kind": "compaction_llm",
        "prompt": prompt,
        "response": response,
        "action": action,
        "kept_recent": kept_recent,
    }
    if usage:
        event.update(usage)
    append_event(thread_id, event)


def _fallback_summary(serialized: str) -> str:
    excerpt = serialized[:_FALLBACK_SUMMARY_CHARS]
    if len(serialized) > _FALLBACK_SUMMARY_CHARS:
        excerpt += "\n...[truncated]"
    return "[Compaction summary unavailable]\n" + excerpt


def _model_name(model) -> str:
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return "unknown"


# --- tool output compaction ---
def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    # find ToolMessage and summarize the content
    if not isinstance(message, ToolMessage):
        return message

    content = _summarize_tool_output(str(message.content), message.name or "")
    return message.model_copy(update={"content": content})


def _summarize_tool_output(content: str, tool_name: str = "") -> str:
    # if the content is small (<= 900 chars and <= 20 lines), return the content
    lines = content.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if len(content) <= _SMALL_CHARS and len(nonempty) <= _SMALL_LINES:
        return content

    # otherwise, summarize the content; get the head, tail and error lines (by regex search)
    head = nonempty[:_HEAD_LINES]
    tail = nonempty[-_TAIL_LINES:] if len(nonempty) > _TAIL_LINES else []
    errors = [ln for ln in nonempty if _ERROR_RE.search(ln)][:_MAX_ERROR_LINES]
    kept = _ordered_dedupe(head, errors, tail)
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

def _ordered_dedupe(*blocks: list[str]) -> list[str]:
    # eleminate duplicate lines with seen set
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        for line in block:
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
    return out

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


# --- compaction entry point ---
def _partition_messages(messages: list[BaseMessage]) -> tuple[list[BaseMessage], list[BaseMessage]]:
    # helper to split the messages into system and non-system messages
    system = [message for message in messages if message.type == "system"]
    rest = [message for message in messages if message.type != "system"]
    return system, rest


async def compact_messages_progressively(
    messages: list[BaseMessage],
    max_tokens: int | None = None,
    *,
    known_input_tokens: int | None = None,
    summary_model=None,
    force: bool = False,
    model_name: str | None = None,
    thread_id: str | None = None,
) -> CompactionResult:
    """Compact by model-relative pressure while preserving the legacy call shape."""
    # calculate the context pressure
    pressure = calculate_context_pressure(
        messages,
        max_tokens=max_tokens,
        known_input_tokens=known_input_tokens,
        model_name=model_name,
    )
    # get the compaction action and how many recent messages to keep
    action = pressure.action
    keep_recent = pressure.keep_recent
    # partition the messages into system and non-system messages
    system, rest_raw = _partition_messages(messages)
    # if force is True, apply the force floor
    if force:
        action, keep_recent = apply_force_floor(action, keep_recent, len(rest_raw))

    if action == "none":
        return CompactionResult(
            messages=messages,
            compacted=False,
            token_count=pressure.token_count,
            action=action,
            kept_recent=keep_recent,
            pressure_ratio=pressure.ratio,
            usable_budget=pressure.usable_budget,
        )

    rest = [_compact_tool_message(message) for message in rest_raw]

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
    keep = min(len(rest), keep_recent)
    older = rest[: len(rest) - keep]
    summary = await summarize_history(
        older,
        summary_model,
        thread_id=thread_id,
        action=action,
        kept_recent=keep,
    )
    # add the system messages and the summary
    compacted = list(system)
    compacted.append(SystemMessage(content="COMPACTED HISTORY\n" + summary))
    # add the recent messages
    compacted.extend(rest[-keep:] if keep else [])
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


def format_compaction_overlay_note(
    result: CompactionResult,
    *,
    budget: int | None = None,
    had_stored_compaction: bool = False,
    warn_ratio: float = 0.7,
) -> str:
    """Build a short L3 note about compaction state and approaching pressure."""
    parts: list[str] = []
    usable_budget = result.usable_budget or budget or settings.compaction_token_budget
    ratio = result.pressure_ratio
    if ratio <= 0 and usable_budget > 0:
        ratio = result.token_count / usable_budget

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