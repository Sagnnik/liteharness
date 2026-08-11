from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from ness_cli.provider.codex.auth import CodexAuth

CODEX_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
"""
type:
    response.completed: final metadata about the whole generation

    response.output_text.delta: pieces of generated text

    response.output_item.done: finished structured outputs

    response.failed: error details
"""


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

    def __init__(self, auth: CodexAuth, *, max_retries: int = 3, timeout: float = 180) -> None:
        self.auth = auth
        self.max_retries = max_retries
        self.timeout = timeout

    async def create(self, payload: dict[str, Any]) -> dict[str, Any]:
        refreshed_401 = False  # needs auth refresh
        retry_attempt = 0 # retry up to max_retries times
        while True:
            credentials = await self.auth.valid_credentials()
            headers = {
                "Authorization": f"Bearer {credentials.access_token}",
                "ChatGPT-Account-ID": credentials.account_id,
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "OpenAI-Beta": "responses=experimental", # extra header for experimental features
                "originator": "ness-agent",
            }
            try:
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    # stream from the Codex Responses URL
                    async with client.stream(
                        "POST", 
                        CODEX_RESPONSES_URL, 
                        headers=headers, 
                        json={**payload, "stream": True}
                    ) as response:
                        # handle 401 Unauthorized: likely due to token expiration
                        if response.status_code == 401 and not refreshed_401:
                            await response.aread() # read the response body once
                            refreshed_401 = True # can refresh only once
                            await self.auth.valid_credentials(force_refresh=True)
                            continue

                        response.raise_for_status()
                        completed: dict[str, Any] | None = None
                        output: list[dict[str, Any]] = []
                        text_parts: list[str] = []
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"): # ignore non-data lines
                                continue
                            raw = line[5:].strip() # remove the "data:" prefix
                            
                            if not raw or raw == "[DONE]": # ignore empty or done lines
                                continue
                            
                            event = json.loads(raw) # parse the JSON event
                            kind = event.get("type") # get the event type
                            
                            # terminal response
                            if kind == "response.completed":
                                value = event.get("response")
                                if isinstance(value, dict):
                                    completed = value # store the completed response

                            # one fully completed output item
                            elif kind == "response.output_item.done":
                                item = event.get("item")
                                if isinstance(item, dict):
                                    output.append(item) # add to the output list

                            # streaming text delta
                            elif kind == "response.output_text.delta":
                                text_parts.append(str(event.get("delta") or "")) # gather the chunks

                            # final text chunk (fallback)
                            elif kind == "response.output_text.done" and not text_parts:
                                text_parts.append(str(event.get("text") or ""))

                            # error: HTTP is 200 but un-successful LLM response
                            elif kind in {"response.failed", "error"}:
                                raise RuntimeError(str(event.get("error") or event))
                        return merge_streamed_response(completed, output, text_parts) # combine all of them into one complete response

            except (httpx.TransportError, httpx.HTTPStatusError, json.JSONDecodeError) as exc:
                retryable = isinstance(exc, httpx.TransportError)
                if isinstance(exc, httpx.HTTPStatusError):
                    retryable = exc.response.status_code == 429 or exc.response.status_code >= 500
                if not retryable or retry_attempt >= self.max_retries:
                    raise
                await asyncio.sleep(min(2**retry_attempt, 8)) # exponential backoff up to 8 seconds
                retry_attempt += 1
