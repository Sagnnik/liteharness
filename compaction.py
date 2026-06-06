from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage

from config import cost_tracker, settings
from prompt import build_compaction_prompt

_SMALL_CHARS = 900
_SMALL_LINES = 20
_MAX_BODY_CHARS = 2400
_HEAD_LINES = 8
_TAIL_LINES = 4
_MAX_ERROR_LINES = 12
_SUMMARY_MESSAGE_LIMIT = 40
_SUMMARY_MESSAGE_CHARS = 1200
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
    0: <8k (No compaction)
    1: <16k (Compact tool outputs)
    2: <32k (Compact tool outputs and recent decisions)
    3: <64k (Compact tool outputs and recent decisions and blockers)
    4: >64k (Full summary)
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
            messages=_dedupe_messages(compacted),
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
    older = rest[: max(0, len(rest) - len(keep))]
    summary = await summarize_history(older, summary_model)
    compacted = system
    if summary:
        compacted.append(SystemMessage(content="COMPACTED HISTORY\n" + summary))
    compacted.extend(_compact_tool_message(msg) for msg in keep)
    return CompactionResult(
        messages=_dedupe_messages(compacted),
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
    if model is None:
        return ""
    serialized = _messages_to_text(messages)
    if not serialized.strip():
        return ""
    prompt = build_compaction_prompt(serialized)
    try:
        response = await model.ainvoke([HumanMessage(content=prompt)])
    except Exception:
        return ""
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
    return summary


def _model_name(model) -> str:
    for attr in ("model_name", "model"):
        value = getattr(model, attr, None)
        if value:
            return str(value)
    return settings.model_name


def _messages_to_text(messages: Iterable[BaseMessage]) -> str:
    items = list(messages)[-_SUMMARY_MESSAGE_LIMIT:]
    return "\n\n".join(
        f"{msg.type}: {_content_text(msg.content)[:_SUMMARY_MESSAGE_CHARS]}" for msg in items
    )


def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, ToolMessage):
        return message

    name = getattr(message, "name", None) or ""
    content = _summarize_tool_output(str(message.content), name)
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


def _content_text(content) -> str:
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("type") or item))
            else:
                parts.append(str(item))
        return " ".join(parts)
    return str(content)


def _dedupe_messages(messages: Iterable[BaseMessage]) -> list[BaseMessage]:
    seen: set[int] = set()
    out = []
    for msg in messages:
        if id(msg) in seen:
            continue
        seen.add(id(msg))
        out.append(msg)
    return out
