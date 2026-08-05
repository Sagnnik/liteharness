from __future__ import annotations

"""Cache-safe conversation summarisation.

``summarize`` is intentionally the only public API in this module.  Callers
must pass the exact parent request messages and the already-bound parent
model.  The function appends one human instruction, preserving the complete
parent prefix for provider prompt caches.
"""

from collections.abc import Sequence
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from ness_agent.instructions import COMPACTION

__all__ = ["summarize"]


class _SummaryInvocationError(RuntimeError):
    pass


async def _invoke_summary(
    messages: Sequence[BaseMessage],
    model: Any,
    *,
    instruction: str,
    max_output_tokens: int,
):
    if not messages:
        raise ValueError("summarize() requires the exact parent request messages")
    if model is None:
        raise ValueError("summarize() requires the already-bound parent model")
    if max_output_tokens <= 0:
        raise ValueError("max_output_tokens must be positive")
    prompt = (instruction or "").strip()
    if not prompt:
        raise ValueError("compaction instruction must not be empty")

    request = [*messages, HumanMessage(content=prompt)]
    try:
        response = await model.ainvoke(request, max_tokens=max_output_tokens)
    except TypeError as exc:
        # A few minimal/fake LangChain models do not accept invocation kwargs.
        # Real provider models do; retain SDK compatibility without changing
        # the cache-safe request shape.
        if "max_tokens" not in str(exc):
            raise
        response = await model.ainvoke(request)

    if getattr(response, "tool_calls", None):
        raise _SummaryInvocationError("compaction model attempted a tool call")
    text = str(getattr(response, "content", "") or "").strip()
    if not text:
        raise _SummaryInvocationError("compaction model returned an empty summary")
    return text, response, request


async def summarize(
    messages: Sequence[BaseMessage],
    model: Any,
    *,
    instruction: str = COMPACTION,
    max_output_tokens: int = 4096,
) -> str:
    """Summarize an exact parent request using a cache-safe human tail.

    ``model`` must be the same already-bound runnable used by the parent
    conversation.  In particular, do not construct a tool-less auxiliary
    model: identical system messages and tool definitions are what allow the
    provider to reuse the parent's cached prefix.
    """

    text, _response, _request = await _invoke_summary(
        messages,
        model,
        instruction=instruction,
        max_output_tokens=max_output_tokens,
    )
    return text
