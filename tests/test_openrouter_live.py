from __future__ import annotations

import asyncio
import os

import pytest
from langchain_core.messages import HumanMessage

from ness_cli.chat_model import (
    ModelOverrides,
    build_chat_model,
    configure_model,
)


@pytest.mark.live
@pytest.mark.skipif(
    os.environ.get("OPENROUTER_LIVE_TEST") != "1",
    reason="set OPENROUTER_LIVE_TEST=1 to spend a minimal live request",
)
def test_openrouter_live_chat_smoke() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY is required")
    model_name = os.environ.get(
        "OPENROUTER_LIVE_MODEL",
        "anthropic/claude-haiku-4.5",
    )
    configure_model(
        ModelOverrides(
            model_name=model_name,
            openai_api_key=api_key,
        )
    )
    try:
        model = build_chat_model("live-smoke")
        response = asyncio.run(
            model.ainvoke([HumanMessage(content="Reply with exactly: OK")])
        )
        assert "OK" in str(response.content)
    finally:
        configure_model(None)
