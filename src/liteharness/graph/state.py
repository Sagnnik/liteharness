from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    approval_declined: bool
    todos: list[dict]
    agent_mode: str
    activate_skills: list[str]  # for cli /skill command
    loaded_skills: list[dict] # for L3 overlay loaded skills info
    last_reflected_message_index: int
    compacted_messages: list[BaseMessage]
    compaction_message_count: int
    force_compact: bool
    last_input_tokens: int
    mode_switch: str
    current_user_seq: int | None