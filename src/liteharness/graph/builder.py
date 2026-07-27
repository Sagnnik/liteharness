from __future__ import annotations

from typing import Any, Mapping

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import Runnable
from liteharness.graph.state import AgentState
from liteharness.graph.nodes import make_nodes


def build_graph(
    config,
    *,
    thread_id,
    mode="act",
    git_available=None,
    checkpointer: BaseCheckpointSaver | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Runnable:
    """Build the graph for the agent"""
    runtime = make_nodes(
        config,
        thread_id=thread_id,
        mode=mode,
        git_available=git_available,
        metadata=metadata,
    )
    graph = StateGraph(AgentState)

    graph.add_node("agent", runtime.agent_node)
    graph.add_node("approval_gate", runtime.approval_gate)
    graph.add_node("tools", runtime.tools_node)

    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        runtime.route_after_agent,
    )
    graph.add_conditional_edges(
        "approval_gate",
        runtime.route_after_approval,
    )
    graph.add_edge("tools", "agent")

    return graph.compile(checkpointer=checkpointer or MemorySaver())
