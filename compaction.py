from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage

from config import settings
from tools.common import SOURCE_FILE_EXTENSIONS

_SMALL_CHARS = 900
_SMALL_LINES = 20
_MAX_BODY_CHARS = 2400

_ERROR_RE = re.compile(
    r"(error|failed|exception|traceback|denied|fatal|panic|cannot|"
    r"command not found|no such file|permission denied|exit code|exit status)",
    re.IGNORECASE,
)
_GREP_MATCH_RE = re.compile(r":\d+:|^Binary file|error", re.IGNORECASE)
_GIT_DIFF_RE = re.compile(r"^(diff |index |--- |\+\+\+ |@@ |[+-](?!\+{2}))|error", re.IGNORECASE)

_FILE_PATH_RE = re.compile(
    rf"(?:(?:\.?/)?[\w.-]+/)+[\w.-]+|[\w.-]+\.(?:{'|'.join(SOURCE_FILE_EXTENSIONS)})",
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
    model=None,
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
    # Step 1: Count the tokens and decide the tier
    budget = max_tokens or settings.compaction_token_budget or settings.max_tokens
    token_count = count_message_tokens(messages, model=model)
    # auto trigger = budget * safety margin
    trigger = int(budget * settings.compaction_safety_margin)
    tier = compaction_tier(token_count)
    if not force and token_count <= trigger:
        return CompactionResult(messages=messages, compacted=False, tier=tier, token_count=token_count)

    # Step 2: Compact the messages
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
    summary = build_structured_history_summary(older)
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


def count_message_tokens(messages: list[BaseMessage], model=None) -> int:
    # use the model.get_num_tokens_from_messages() if available
    if model is not None and hasattr(model, "get_num_tokens_from_messages"):
        try:
            return int(model.get_num_tokens_from_messages(messages))
        except Exception:
            pass
    # else build a text and use tiktoken for rough estimate
    text = "\n\n".join(f"{m.type}: {_content_text(m.content)}" for m in messages)
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text)) + 4 * len(messages)
    except Exception:
        # fallback to a rough estimate= sum of all chars // 3 and symbols // 2 and 6 * message count
        symbol_count = sum(1 for char in text if not char.isalnum() and not char.isspace())
        return max(1, len(text) // 3 + symbol_count // 2 + 6 * len(messages))


def compaction_tier(token_count: int) -> int:
    # tier 1 : <8k
    if token_count <= settings.compaction_tier_1:
        return 0
    # tier 2 : <16k
    if token_count <= settings.compaction_tier_2:
        return 1
    # tier 3 : <32k
    if token_count <= settings.compaction_tier_3:
        return 2
    # tier 4 : <64k
    if token_count <= settings.compaction_tier_4:
        return 3
    # tier 5 : >64k
    return 4


def build_structured_history_summary(messages: Iterable[BaseMessage]) -> str:
    decisions: list[str] = []
    blockers: list[str] = []
    files: set[str] = set()
    preferences: list[str] = []
    recent_requests: list[str] = []
    tool_notes: list[str] = []
    for msg in messages:
        text = _content_text(msg.content)
        lower = text.lower()
        if msg.type == "human":
            recent_requests.append(_shorten(text, 260))
            if any(word in lower for word in ("prefer", "please", "don't", "do not", "always", "never")):
                preferences.append(_shorten(text, 220))
        if msg.type in {"ai", "assistant"} and any(
            word in lower for word in ("decided", "decision", "implemented", "changed", "use ")
        ):
            decisions.append(_shorten(text, 260))
        if any(word in lower for word in ("blocker", "blocked", "failed", "error", "cannot", "denied")):
            blockers.append(_shorten(text, 240))
        if msg.type == "tool":
            tool_notes.append(_shorten(text, 220))
        files.update(_extract_file_paths(text))

    sections = []
    if decisions:
        sections.append("Decisions and changes:\n" + "\n".join(f"- {item}" for item in _unique(decisions, 8)))
    if blockers:
        sections.append("Blockers and unresolved issues:\n" + "\n".join(f"- {item}" for item in _unique(blockers, 8)))
    if files:
        sections.append("Files mentioned or modified:\n" + "\n".join(f"- {item}" for item in sorted(files)[:30]))
    if preferences:
        sections.append("User preferences:\n" + "\n".join(f"- {item}" for item in _unique(preferences, 6)))
    if recent_requests:
        sections.append("Earlier user requests:\n" + "\n".join(f"- {item}" for item in _unique(recent_requests, 8)))
    if tool_notes:
        sections.append("Tool results summarized:\n" + "\n".join(f"- {item}" for item in _unique(tool_notes, 8)))
    return "\n\n".join(sections)


def _compact_tool_message(message: BaseMessage) -> BaseMessage:
    if not isinstance(message, ToolMessage):
        return message

    name = getattr(message, "name", None) or ""
    content = _summarize_tool_output(str(message.content), name)
    return message.model_copy(update={"content": content})


def _summarize_tool_output(content: str, tool_name: str = "") -> str:
    # if the content is small, return the content as is
    lines = content.splitlines()
    nonempty = [ln for ln in lines if ln.strip()]
    if len(content) <= _SMALL_CHARS and len(nonempty) <= _SMALL_LINES:
        return content

    # keep the lines that are important
    kept = _lines_to_keep(nonempty, tool_name)
    if len(kept) >= len(nonempty):
        return content

    # truncate the body if it's too long
    body = "\n".join(kept)
    if len(body) > _MAX_BODY_CHARS:
        body = body[:_MAX_BODY_CHARS] + "\n...[truncated]"
    return (
        f"[compacted {tool_name or 'tool'} output] "
        f"{len(nonempty)} lines, {len(content)} chars\n"
        f"{body}"
    )


def _lines_to_keep(lines: list[str], tool_name: str) -> list[str]:
    name = tool_name.lower()

    if name in {"bash", "spawn_subagent"}:
        return _merge_slices(
            lines,
            _matching(lines, _ERROR_RE, limit=12),
            lines[:6],
            lines[-4:],
            max_total=28,
        )

    if name == "grep":
        return _merge_slices(
            lines,
            _matching(lines, _GREP_MATCH_RE, limit=40),
            _matching(lines, _ERROR_RE, limit=8),
            lines[:3],
            max_total=45,
        )

    if name in {"read_file", "glob_files", "list_files", "git_diff", "git_show", "git_blame", "git_log"}:
        important = lines
        if name.startswith("git_") or name == "git_diff":
            important = _matching(lines, _GIT_DIFF_RE, limit=30)
        return _merge_slices(
            lines,
            important,
            lines[:12],
            lines[-4:],
            max_total=32,
        )

    if name.startswith("git_") or name.startswith("mcp__"):
        return _merge_slices(
            lines,
            _matching(lines, _ERROR_RE, limit=10),
            lines[:8],
            lines[-4:],
            max_total=28,
        )

    return _merge_slices(
        lines,
        _matching(lines, _ERROR_RE, limit=12),
        lines[:8],
        lines[-4:],
        max_total=28,
    )


def _matching(lines: list[str], pattern: re.Pattern, *, limit: int) -> list[str]:
    out: list[str] = []
    for line in lines:
        if pattern.search(line):
            out.append(line)
            if len(out) >= limit:
                break
    return out


def _merge_slices(lines: list[str], *slices: list[str], max_total: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for block in slices:
        for line in block:
            if line in seen:
                continue
            seen.add(line)
            out.append(line)
            if len(out) >= max_total:
                return out
    return out if out else lines[:max_total]


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


def _shorten(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _unique(items: Iterable[str], limit: int) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
        if len(out) >= limit:
            break
    return out


def _extract_file_paths(text: str) -> set[str]:
    paths = set()
    for match in _FILE_PATH_RE.findall(text):
        normalized = match.strip("`'\"()[]{}:,")
        if normalized and not normalized.startswith(("http://", "https://")):
            paths.add(normalized)
    return paths
