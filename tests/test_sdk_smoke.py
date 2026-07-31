from __future__ import annotations

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.tools import tool


def test_import_ness_ai_public_api():
    import ness_ai as lh

    for name in (
        "NessAgent",
        "NessAgentConfig",
        "AgentSpec",
        "Session",
        "ContextPreview",
        "NoopTracer",
        "CostTracker",
        "aggregate_usage",
        "PromptLayers",
        "ToolRegistry",
        "coding_tools",
        "NessAgentOptions",
        "message_to_text",
    ):
        assert hasattr(lh, name), name


def test_ness_agent_session_builds_graph():
    from ness_ai import NessAgent, PromptLayers, PromptLayersConfig

    @tool
    def ping() -> str:
        """Return pong."""
        return "pong"

    model = FakeListChatModel(responses=["hello"])
    agent = NessAgent(
        model=model,
        tools=[ping],
        prompt=PromptLayers(PromptLayersConfig(l0="L0", persona="P")),
    )
    session = agent.session(thread_id="t-smoke")
    assert session.app is not None
    assert session.thread_id == "t-smoke"
    assert agent.config.cost_tracker is not None
    assert agent.config.tracer is not None
