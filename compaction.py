from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import cost_tracker, settings
from prompt import _content_text, _messages_to_text as format_messages, build_compaction_prompt

_SMALL_CHARS = 900
_SMALL_LINES = 20
_MAX_BODY_CHARS = 2400
_HEAD_LINES = 8
_TAIL_LINES = 4
_MAX_ERROR_LINES = 12
_SUMMARY_MESSAGE_LIMIT = 40
_FALLBACK_SUMMARY_CHARS = 8000
COMPACTION_SAFETY_MARGIN = 0.80
COMPACTION_TIERS = (8_000, 16_000, 32_000, 64_000)

_ERROR_RE = re.compile(
    r"(error|failed|exception|traceback|denied|fatal|panic|cannot|"
    r"command not found|no such file|permission denied|exit code|exit status)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CompactionResult:
    messages: list[BaseMessage]
    compacted: bool
    tier: int
    token_count: int
    summary: str = ""


async def compact_messages_progressively(
    messages: list[BaseMessage],
    max_tokens: int | None = None,
    *,
    known_input_tokens: int | None = None,
    summary_model=None,
    force: bool = False,
) -> CompactionResult:
    """Compact by token tier.
    Tiers:
    0: <8k (no compaction)
    1: <16k (compact tool outputs)
    2: <32k (compact tool outputs + summarize older history, keep last 10)
    3: <64k (same, keep last 6)
    4: >64k (same, keep last 4)
    """
    budget = max_tokens or settings.compaction_token_budget
    token_count = resolve_token_count(messages, known_input_tokens=known_input_tokens)
    trigger = int(budget * COMPACTION_SAFETY_MARGIN)
    tier = compaction_tier(token_count)
    if not force and token_count <= trigger:
        return CompactionResult(messages=messages, compacted=False, tier=tier, token_count=token_count)

    system = [m for m in messages if m.type == "system"]
    rest = [m for m in messages if m.type != "system"]
    if tier <= 1:
        compacted = system + [_compact_tool_message(msg) for msg in rest]
        return CompactionResult(
            messages=compacted,
            compacted=True,
            tier=1,
            token_count=token_count,
            summary="Summarized tool outputs.",
        )

    if tier == 2:
        keep = rest[-10:]
    elif tier == 3:
        keep = rest[-6:]
    else:
        keep = rest[-4:]
    older = rest[: len(rest) - len(keep)]
    summary = await summarize_history(older, summary_model)
    compacted = system
    if summary:
        compacted.append(SystemMessage(content="COMPACTED HISTORY\n" + summary))
    compacted.extend(_compact_tool_message(msg) for msg in keep)
    return CompactionResult(
        messages=compacted,
        compacted=True,
        tier=tier,
        token_count=token_count,
        summary=summary,
    )


def resolve_token_count(messages: list[BaseMessage], *, known_input_tokens: int | None) -> int:
    if known_input_tokens and known_input_tokens > 0:
        return known_input_tokens
    return _estimate_tokens_symbol(messages)


def _estimate_tokens_symbol(messages: list[BaseMessage]) -> int:
    text = "\n\n".join(f"{m.type}: {_content_text(m.content)}" for m in messages)
    symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
    return max(1, len(text) // 3 + symbol_count // 2 + 6 * len(messages))


def compaction_tier(token_count: int) -> int:
    for index, boundary in enumerate(COMPACTION_TIERS):
        if token_count <= boundary:
            return index
    return len(COMPACTION_TIERS)


async def summarize_history(messages: Iterable[BaseMessage], model) -> str:
    serialized = _serialize_messages(messages)
    if not serialized.strip():
        return ""
    if model is None:
        return _fallback_summary(serialized)
    prompt = build_compaction_prompt(serialized)
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception:
        return _fallback_summary(serialized)
    content = response.content
    if not isinstance(content, str):
        content = str(content)
    summary = content.strip()
    if response.usage_metadata:
        cost_tracker.add(
            response.usage_metadata,
            _model_name(model),
            response.response_metadata or {},
        )
    if summary:
        return summary
    return _fallback_summary(serialized)


def _serialize_messages(messages: Iterable[BaseMessage]) -> str:
    return format_messages(messages, limit=_SUMMARY_MESSAGE_LIMIT)


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


def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, ToolMessage):
        return message

    content = _summarize_tool_output(str(message.content), message.name or "")
    return message.model_copy(update={"content": content})


def _summarize_tool_output(content: str, tool_name: str = "") -> str:
    lines = content.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if len(content) <= _SMALL_CHARS and len(nonempty) <= _SMALL_LINES:
        return content

    head = nonempty[:_HEAD_LINES]
    tail = nonempty[-_TAIL_LINES:] if len(nonempty) > _TAIL_LINES else []
    errors = [ln for ln in nonempty if _ERROR_RE.search(ln)][:_MAX_ERROR_LINES]
    kept = _ordered_dedupe(head, errors, tail)
    if len(kept) >= len(nonempty):
        return content

    body = "\n".join(kept)
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n...[truncated]"
    return (
        f"[compacted {tool_name or 'tool'} output] "
        f"{len(nonempty)} lines, {len(content)} chars\n"
        f"{body}"
    )


def _ordered_dedupe(*blocks: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in blocks:
        for line in block:
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
    return out
