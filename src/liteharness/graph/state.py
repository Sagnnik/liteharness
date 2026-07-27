from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    """Checkpointed graph state for one thread."""
    messages: Annotated[list[BaseMessage], add_messages]
    approval_declined: bool
    todos: list[dict]
    mode: str
    activate_skills: list[str]  
    loaded_skills: list[dict]
    last_reflection_index: int
    compacted_messages: list[BaseMessage]
    compaction_message_count: int
    force_compact: bool
    last_input_tokens: int
    mode_switch: str
