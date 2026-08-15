from __future__ import annotations

import asyncio
import json
import math
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from random import uniform
from typing import Any

import httpx

from ness_cli.provider.codex.auth import CodexAuth

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
_MAX_BACKOFF_SECONDS = 8.0
_MAX_RETRY_DELAY_SECONDS = 60.0
_RETRYABLE_STREAM_ERROR_TYPES = {
    "rate_limit_error",
    "server_error",
    "service_unavailable_error",
}
_RETRYABLE_STREAM_ERROR_CODES = {
    "internal_server_error",
    "rate_limit_exceeded",
    "server_error",
    "server_is_overloaded",
}


class CodexStreamError(RuntimeError):
    """A terminal error delivered inside an otherwise successful SSE response."""

    def __init__(
        self,
        error: Any,
        *,
        retryable: bool,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(str(error))
        self.retryable = retryable
        self.retry_after = retry_after


async def _sleep(delay: float) -> None:
    await asyncio.sleep(delay)


def _retry_after_seconds(headers: httpx.Headers) -> float | None:
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        delay = float(raw)
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
    return delay if math.isfinite(delay) and delay >= 0 else None


def _retry_delay(retry_attempt: int, retry_after: float | None) -> float:
    base = (
        retry_after
        if retry_after is not None
        else min(2**retry_attempt, _MAX_BACKOFF_SECONDS)
    )
    jitter = uniform(0, min(1.0, max(base, 0.0) * 0.25))
    return min(base + jitter, _MAX_RETRY_DELAY_SECONDS)


def _stream_error(event: dict[str, Any], headers: httpx.Headers) -> CodexStreamError:
    error: Any = event.get("error")
    if error is None:
        response = event.get("response")
        if isinstance(response, dict):
            error = response.get("error")
    if error is None:
        error = event

    details = error if isinstance(error, dict) else event
    error_type = str(details.get("type") or "").casefold()
    error_code = str(details.get("code") or "").casefold()
    retryable = (
        error_type in _RETRYABLE_STREAM_ERROR_TYPES
        or error_code in _RETRYABLE_STREAM_ERROR_CODES
    )
    return CodexStreamError(
        error,
        retryable=retryable,
        retry_after=_retry_after_seconds(headers),
    )


def merge_streamed_response(
    completed: dict[str, Any] | None,
    output_items: list[dict[str, Any]],
    text_parts: list[str],
) -> dict[str, Any]:
    """Merge streamed content into the terminal response envelope.

    The ChatGPT Codex backend can emit a metadata/usage-only
    ``response.completed`` envelope. Returning that envelope verbatim drops
    the already-received text deltas, producing an empty AIMessage even though
    output-token usage is non-zero.
    """
    response = dict(completed or {})
    if output_items:
        # output_item.done contains complete replayable items (message,
        # reasoning, and function calls), so prefer it to a sparse envelope.
        response["output"] = [dict(item) for item in output_items]
    streamed_text = "".join(text_parts)
    if streamed_text:
        response["output_text"] = streamed_text
    return response


class CodexResponsesTransport:
    """Experimental, deliberately isolated ChatGPT Codex Responses transport."""

    def __init__(
        self, auth: CodexAuth, *, max_retries: int = 3, timeout: float = 180
    ) -> None:
        self.auth = auth
        self.max_retries = max_retries
        self.timeout = timeout

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        refreshed_401 = False  # needs auth refresh
        retry_attempt = 0  # retry up to max_retries times
        while True:
            credentials = await self.auth.valid_credentials()
            headers = {
                "Authorization": f"Bearer {credentials.access_token}",
                "ChatGPT-Account-ID": credentials.account_id,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "OpenAI-Beta": "responses=experimental",  # extra header for experimental features
                "originator": "ness-agent",
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # stream from the Codex Responses URL
                    async with client.stream(
                        "POST",
                        CODEX_RESPONSES_URL,
                        headers=headers,
                        json={**payload, "stream": True},
                    ) as response:
                        # handle 401 Unauthorized: likely due to token expiration
                        if response.status_code == 401 and not refreshed_401:
                            await response.aread()  # read the response body once
                            refreshed_401 = True  # can refresh only once
                            await self.auth.valid_credentials(force_refresh=True)
                            continue

                        response.raise_for_status()
                        completed: dict[str, Any] | None = None
                        output: list[dict[str, Any]] = []
                        text_parts: list[str] = []
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):  # ignore non-data lines
                                continue
                            raw = line[5:].strip()  # remove the "data:" prefix

                            if not raw or raw == "[DONE]":  # ignore empty or done lines
                                continue

                            event = json.loads(raw)  # parse the JSON event
                            kind = event.get("type")  # get the event type

                            # terminal response
                            if kind == "response.completed":
                                value = event.get("response")
                                if isinstance(value, dict):
                                    completed = value  # store the completed response

                            # one fully completed output item
                            elif kind == "response.output_item.done":
                                item = event.get("item")
                                if isinstance(item, dict):
                                    output.append(item)  # add to the output list

                            # streaming text delta
                            elif kind == "response.output_text.delta":
                                text_parts.append(
                                    str(event.get("delta") or "")
                                )  # gather the chunks

                            # final text chunk (fallback)
                            elif kind == "response.output_text.done" and not text_parts:
                                text_parts.append(str(event.get("text") or ""))

                            # error: HTTP is 200 but un-successful LLM response
                            elif kind in {"response.failed", "error"}:
                                raise _stream_error(event, response.headers)
                        return merge_streamed_response(
                            completed, output, text_parts
                        )  # combine all of them into one complete response

            except (
                CodexStreamError,
                httpx.TransportError,
                httpx.HTTPStatusError,
                json.JSONDecodeError,
            ) as exc:
                retryable = isinstance(exc, httpx.TransportError)
                if isinstance(exc, httpx.HTTPStatusError):
                    retryable = (
                        exc.response.status_code == 429
                        or exc.response.status_code >= 500
                    )
                elif isinstance(exc, CodexStreamError):
                    retryable = exc.retryable
                if not retryable or retry_attempt >= self.max_retries:
                    raise
                retry_after = (
                    exc.retry_after
                    if isinstance(exc, CodexStreamError)
                    else (
                        _retry_after_seconds(exc.response.headers)
                        if isinstance(exc, httpx.HTTPStatusError)
                        else None
                    )
                )
                await _sleep(_retry_delay(retry_attempt, retry_after))
                retry_attempt += 1
