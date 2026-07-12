from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool

from liteharness import NessAgent, PromptLayers, PromptLayersConfig
from liteharness.compaction import CompactionResult, apply_force_floor
from liteharness.context.layers import TaskPrompts
from liteharness.graph.nodes import make_nodes
from liteharness.options import NessAgentOptions
from liteharness.persistence import ThreadStore
from liteharness.reflection import finalize_session_reflection, run_reflection_gate
from liteharness.usage import CostTracker


def _agent(tmp_path: Path, **kwargs):
    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    return NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        **kwargs,
    )


def test_apply_force_floor_upgrades_to_summary_when_history_long():
    assert apply_force_floor("none", 0, 11) == ("summary", 10)
    assert apply_force_floor("tool_outputs", 0, 11) == ("summary", 10)
    assert apply_force_floor("summary", 5, 11) == ("summary", 5)


def test_apply_force_floor_tool_outputs_when_history_short():
    assert apply_force_floor("none", 0, 10) == ("tool_outputs", 0)
    assert apply_force_floor("none", 0, 5) == ("tool_outputs", 0)
    assert apply_force_floor("tool_outputs", 3, 5) == ("tool_outputs", 3)


def test_cost_tracker_uses_estimate_when_no_provider_cost():
    tracker = CostTracker(estimate_cost=lambda model, u, c, o: 1.25)
    usage = tracker.add(
        {"input_tokens": 100, "output_tokens": 20},
        model_name="test-model",
        response_metadata={},
    )
    assert usage is not None
    assert usage["cost_usd"] == 1.25
    assert usage["cost_source"] == "estimated"


def test_cost_tracker_prefers_provider_cost():
    tracker = CostTracker(estimate_cost=lambda model, u, c, o: 9.99)
    usage = tracker.add(
        {"input_tokens": 10, "output_tokens": 2},
        model_name="test-model",
        response_metadata={"cost": 0.01},
    )
    assert usage is not None
    assert usage["cost_usd"] == 0.01
    assert usage["cost_source"] == "provider"


def test_make_sdk_cost_tracker_wires_cli_pricing():
    from liteharness_cli.config import make_sdk_cost_tracker

    tracker = make_sdk_cost_tracker()
    assert tracker.estimate_cost is not None


def test_thread_store_records_default_model(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="gpt-test")
    store.append_event("session-a", {"kind": "user", "content": "hi"})
    rows = store.list_threads()
    assert rows[0]["model"] == "gpt-test"


def test_thread_store_backfills_model_from_usage(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="")
    store.append_event("session-b", {"kind": "user", "content": "hi"})
    assert store.list_threads()[0]["model"] == ""
    store.append_event(
        "session-b",
        {"kind": "usage", "model": "deepseek-v4", "input_tokens": 1, "output_tokens": 1},
    )
    assert store.list_threads()[0]["model"] == "deepseek-v4"


def test_ness_agent_wires_thread_store_default_model(tmp_path: Path):
    model = FakeListChatModel(responses=["ok"])
    object.__setattr__(model, "model_name", "wired-model")

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    assert agent.config.thread_store.default_model == "wired-model"


def test_reflection_error_event_has_full_schema(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")

    class BoomModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, _messages):
            raise RuntimeError("boom")

    result = asyncio.run(
        run_reflection_gate(
            "session-r",
            [HumanMessage(content="hello")],
            BoomModel(),
            1,
            persistence=store,
            task_prompts=TaskPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert result.error == "boom"
    events = store.load_thread_events("session-r")
    reflection = next(e for e in events if e.get("kind") == "reflection")
    assert set(reflection) >= {
        "kind",
        "prompt",
        "response",
        "message_index",
        "memory_updated",
        "error",
        "t",
    }
    assert reflection["error"] == "boom"
    assert reflection["response"] == {"new_bullet_points": []}
    assert reflection["memory_updated"] is False
    assert "No todos" in reflection["prompt"]


def test_finalize_session_reflection_passes_rendered_todos(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    captured: dict = {}

    class CaptureModel:
        def with_structured_output(self, _schema):
            return self

        async def ainvoke(self, messages):
            captured["prompt"] = messages[0].content
            return SimpleNamespace(
                model_dump=lambda: {"new_bullet_points": []},
                new_bullet_points=[],
            )

    app = MagicMock()
    app.aget_state = AsyncMock(
        return_value=SimpleNamespace(
            values={
                "messages": [HumanMessage(content="do work")],
                "todos": [{"id": "1", "content": "Ship it", "status": "pending"}],
                "last_reflected_message_index": 0,
            }
        )
    )

    asyncio.run(
        finalize_session_reflection(
            app,
            "session-f",
            CaptureModel(),
            persistence=store,
            task_prompts=TaskPrompts(
                reflection=(
                    "t={thread_id} n={user_message_count} msgs={messages} "
                    "bullets={current_session_bullets} todos={todos}"
                )
            ),
        )
    )
    assert "[pending] 1: Ship it" in captured["prompt"]


def test_agent_node_sets_last_input_tokens_zero_after_compaction(tmp_path: Path):
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="m")
    agent = _agent(tmp_path)
    agent.config.thread_store = store
    rt = make_nodes(agent.config, thread_id="session-c", agent_mode="act", git_available=False)

    compacted = CompactionResult(
        messages=[HumanMessage(content="hi")],
        compacted=True,
        token_count=10,
        action="summary",
    )
    response = AIMessage(content="ok")

    async def fake_ainvoke(_msgs):
        return response

    with (
        patch(
            "liteharness.graph.nodes.compact_messages_progressively",
            return_value=compacted,
        ),
        patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ),
    ):
        updates = asyncio.run(
            rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "agent_mode": "act",
                    "todos": [],
                    "last_input_tokens": 999,
                }
            )
        )

    assert updates["last_input_tokens"] == 0
    assert updates["compaction_message_count"] == 1


def test_usage_event_always_logged_with_model(tmp_path: Path):
    model = FakeListChatModel(responses=["hello"])
    object.__setattr__(model, "model", "usage-model")
    store = ThreadStore(threads_dir=tmp_path / "threads", default_model="")

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
        cost_tracker=CostTracker(),
    )
    agent.config.thread_store = store
    agent.config.cost_tracker.add = lambda *a, **k: None  # type: ignore[method-assign]
    rt = make_nodes(agent.config, thread_id="session-u", agent_mode="act", git_available=False)

    response = AIMessage(content="ok")
    response.usage_metadata = {"input_tokens": 5, "output_tokens": 1}

    async def fake_ainvoke(_msgs):
        return response

    with (
        patch(
            "liteharness.graph.nodes.compact_messages_progressively",
            return_value=CompactionResult(
                messages=[HumanMessage(content="hi")],
                compacted=False,
                token_count=5,
            ),
        ),
        patch.object(
            agent.config.tool_registry,
            "bind_model",
            return_value=SimpleNamespace(ainvoke=fake_ainvoke),
        ),
    ):
        asyncio.run(
            rt.agent_node(
                {
                    "messages": [HumanMessage(content="hi")],
                    "agent_mode": "act",
                    "todos": [],
                    "last_input_tokens": 0,
                }
            )
        )

    events = agent.config.thread_store.load_thread_events("session-u")
    usage = next(e for e in events if e.get("kind") == "usage")
    assert usage["model"] == "usage-model"


def test_options_context_window_drives_usable_budget():
    from liteharness.compaction import resolve_usable_context_budget

    opts = NessAgentOptions(
        context_window=100_000,
        compaction_output_reserve_tokens=8_000,
        compaction_input_reserve_tokens=2_000,
    )
    assert resolve_usable_context_budget("any-model", opts) == 90_000
    assert resolve_usable_context_budget("any-model", None) == 120_000
    assert resolve_usable_context_budget(
        "any-model",
        NessAgentOptions(context_window=None, compaction_token_budget=50_000),
    ) == 50_000


def test_agent_spec_resolves_backends(tmp_path: Path):
    from liteharness import AgentSpec, NessAgent
    from liteharness.tracing.tracer import NoopTracer

    model = FakeListChatModel(responses=["ok"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    spec = AgentSpec(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(ness_dir=tmp_path / ".ness", project_root=tmp_path),
    )
    agent = NessAgent.from_spec(spec)
    cfg = agent.config
    assert cfg.memory_store is not None
    assert cfg.thread_store is not None
    assert cfg.permission_store is not None
    assert cfg.tool_registry is not None
    assert cfg.cost_tracker is not None
    assert isinstance(cfg.tracer, NoopTracer)
    assert not hasattr(cfg, "budget")
    assert not hasattr(cfg, "permission_policy")
    assert not hasattr(cfg, "mcp_config")


def test_reflection_schedule_budget_resolve_no_typeerror(tmp_path: Path):
    """reflection_token_ratio > 0 must call resolve_usable_context_budget correctly."""
    from liteharness.graph.nodes import _maybe_schedule_reflection, make_nodes

    model = FakeListChatModel(responses=["hello"])

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
        options=NessAgentOptions(
            ness_dir=tmp_path / ".ness",
            project_root=tmp_path,
            reflection_token_ratio=0.5,
            context_window=100_000,
        ),
    )
    rt = make_nodes(agent.config, thread_id="session-ref", agent_mode="act", git_available=False)
    # Should not raise TypeError (previously passed budget as meta).
    _maybe_schedule_reflection(
        rt,
        {"last_reflected_message_index": 0, "todos": []},
        [HumanMessage(content="hello")],
        "test-model",
    )
