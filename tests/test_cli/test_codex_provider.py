from __future__ import annotations

import base64
import asyncio
import json
import time
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from ness_cli.config import settings
from ness_cli.provider.codex.adapter import CodexProviderAdapter
from ness_cli.provider.codex.auth import CodexAuth, _jwt_expiry
from ness_cli.provider.codex.chat_model import CodexSubscriptionChatModel
from ness_cli.provider.codex.transport import merge_streamed_response
from ness_cli.provider.openrouter.adapter import OpenRouterProviderAdapter
from ness_cli.provider.profile import provider_profile, update_provider_profile
from ness_agent.tracing.cost import CostTracker


def _jwt(expiry: int) -> str:
    payload = base64.urlsafe_b64encode(json.dumps({"exp": expiry}).encode()).decode().rstrip("=")
    return f"header.{payload}.signature"


def test_provider_profile_roundtrip_is_namespaced(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    update_provider_profile("codex", {"model_name": "gpt-test", "reasoning_effort": "high"})
    update_provider_profile("openrouter", {"model_name": "openai/gpt-test"})

    assert provider_profile("codex") == {"model_name": "gpt-test", "reasoning_effort": "high"}
    assert provider_profile("openrouter") == {"model_name": "openai/gpt-test"}


def test_codex_auth_reads_only_isolated_home(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "codex"
    home.mkdir()
    expiry = int(time.time()) + 3600
    (home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _jwt(expiry),
                    "account_id": "acct-test",
                    "refresh_token": "secret",
                },
            }
        ),
        encoding="utf-8",
    )

    credentials = CodexAuth().credentials()
    assert credentials is not None
    assert credentials.account_id == "acct-test"
    assert credentials.expires_at == expiry
    assert _jwt_expiry("invalid") is None


def test_codex_chat_model_preserves_function_call_history():
    model = CodexSubscriptionChatModel(model="gpt-test", reasoning_effort="high")
    instructions, items = model._input(
        [
            HumanMessage(content="inspect"),
            AIMessage(
                content="",
                tool_calls=[{"name": "read", "args": {"path": "a.py"}, "id": "call-1", "type": "tool_call"}],
            ),
            ToolMessage(content="contents", tool_call_id="call-1"),
        ]
    )

    assert instructions == ""
    assert items[1] == {
        "type": "function_call",
        "call_id": "call-1",
        "name": "read",
        "arguments": '{"path":"a.py"}',
    }
    assert items[2]["type"] == "function_call_output"


def test_codex_response_maps_usage_tools_and_subscription_billing():
    message = CodexSubscriptionChatModel._message(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "output": [
                {"type": "message", "content": [{"type": "output_text", "text": "checking"}]},
                {"type": "function_call", "call_id": "call-1", "name": "read", "arguments": '{"path":"a.py"}'},
            ],
            "usage": {"input_tokens": 10, "output_tokens": 4, "input_tokens_details": {"cached_tokens": 3}},
        }
    )

    assert message.content == "checking"
    assert message.tool_calls[0]["args"] == {"path": "a.py"}
    assert message.usage_metadata["input_token_details"]["cache_read"] == 3
    assert message.response_metadata["billing_mode"] == "subscription"


def test_sparse_completed_response_keeps_streamed_assistant_text():
    response = merge_streamed_response(
        {
            "id": "resp-1",
            "model": "gpt-test",
            "output": [],
            "usage": {"input_tokens": 10, "output_tokens": 3},
        },
        [{"type": "reasoning", "id": "reasoning-1"}],
        ["hel", "lo"],
    )
    message = CodexSubscriptionChatModel._message(response)

    assert message.content == "hello"
    assert message.usage_metadata["output_tokens"] == 3
    replayable = message.additional_kwargs["codex_output_items"]
    assert any(item.get("type") == "message" for item in replayable)


def test_streamed_output_items_replace_sparse_completed_output():
    streamed = [
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "read",
            "arguments": '{"path":"a.py"}',
        }
    ]
    response = merge_streamed_response({"output": []}, streamed, [])
    message = CodexSubscriptionChatModel._message(response)

    assert message.tool_calls[0]["name"] == "read"


def test_login_readiness_waits_for_account_state(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    home = tmp_path / "codex"
    home.mkdir()
    (home / "auth.json").write_text(
        json.dumps(
            {
                "tokens": {
                    "access_token": _jwt(int(time.time()) + 3600),
                    "account_id": "acct-test",
                }
            }
        ),
        encoding="utf-8",
    )

    class EventuallyReadyServer:
        def __init__(self):
            self.calls = 0

        async def request(self, method, params):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("account state is still converging")
            if self.calls == 2:
                return {"account": None}
            return {"account": {"type": "chatgpt"}}

    server = EventuallyReadyServer()
    credentials = asyncio.run(CodexAuth(server).wait_until_ready(timeout=1))  # type: ignore[arg-type]

    assert credentials.account_id == "acct-test"
    assert server.calls == 3


def test_device_login_explains_chatgpt_security_prerequisite():
    adapter = CodexProviderAdapter()

    device = next(method for method in adapter.login_methods() if method.id == "device")
    message = adapter._login_error(
        "Enable device code authorization for Codex in ChatGPT Security Settings, "
        'then run "codex login --device-auth" again.'
    )

    assert device.guidance is not None
    assert "Settings > Security" in device.guidance
    assert "Device-code authorization is disabled" in message
    assert "Open browser" in message
    assert "/login" in message


def test_device_login_start_translates_security_error():
    adapter = CodexProviderAdapter()

    class DisabledDeviceAuthServer:
        async def start(self):
            return None

        async def request(self, method, params):
            raise RuntimeError(
                "Enable device code authorization for Codex in ChatGPT Security Settings"
            )

    adapter.server = DisabledDeviceAuthServer()  # type: ignore[assignment]
    result = asyncio.run(adapter.login(method="device"))

    assert result.status == "error"
    assert "Settings > Security" in result.message
    assert "Open browser" in result.message


def test_openrouter_adapter_owns_masked_key_login(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AGENT_CONFIG_DIR", str(tmp_path))
    previous = settings.openai_api_key
    adapter = OpenRouterProviderAdapter()
    try:
        method = adapter.login_methods()[0]
        result = asyncio.run(
            adapter.login(method=method.id, secret="  sk-or-test  ")
        )

        assert method.input_kind == "secret"
        assert method.input_label == "OpenRouter API key"
        assert result.status == "complete"
        assert settings.openai_api_key == "sk-or-test"
        assert (tmp_path / "secrets.json").exists()
    finally:
        settings.openai_api_key = previous


def test_subscription_metadata_disables_api_price_estimate():
    tracker = CostTracker(pricing={"gpt-test": (10.0, 20.0, 0.1)})
    usage = tracker.add(
        {"input_tokens": 1_000, "output_tokens": 500},
        "gpt-test",
        {"billing_mode": "subscription"},
    )
    assert usage is not None
    assert usage.cost_usd is None
    assert usage.cost_source is None


def test_codex_status_deduplicates_windows_and_exposes_weekly(monkeypatch):
    adapter = CodexProviderAdapter()

    class FakeAuth:
        def is_authenticated(self):
            return True

    class FakeServer:
        async def start(self):
            return None

        async def request(self, method, params):
            if method == "account/read":
                return {"account": {"type": "chatgpt", "email": "user@example.com", "planType": "plus"}}
            snapshot = {
                "limitId": "codex",
                "limitName": "Codex",
                "primary": {"usedPercent": 20, "windowDurationMins": 300, "resetsAt": 2_000_000_000},
                "secondary": {"usedPercent": 40, "windowDurationMins": 10080, "resetsAt": 2_000_100_000},
            }
            return {
                "rateLimits": snapshot,
                "rateLimitsByLimitId": {"codex": snapshot},
                "rateLimitResetCredits": {"availableCount": 2},
            }

    adapter.auth = FakeAuth()  # type: ignore[assignment]
    adapter.server = FakeServer()  # type: ignore[assignment]
    status = asyncio.run(adapter.status(refresh=True))

    assert status.account.email == "user@example.com"
    assert status.account.tier == "plus"
    assert len(status.limits) == 2
    assert {bucket.window_minutes for bucket in status.limits} == {300, 10080}
    assert status.credits == "2 available"
