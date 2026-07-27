from __future__ import annotations

from typing import Any
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.messages import ToolMessage, AIMessage
from liteharness.compaction import resolve_token_count
from liteharness.context.overlay import wrap_system_reminder
from liteharness.tools import ToolRegistry
from liteharness.graph.state import AgentState

def _effective_conversation(messages, state) -> list[BaseMessage]:
    """
    Build the effective message list at every turn
    if compaction exists then we need compacted + raw[source_count:] else just raw system message
    """
    compacted = list(state.get("compacted_messages", []))
    source_count = int(state.get("compaction_message_count", 0) or 0)
    raw = [m for m in messages if m.type != "system"]
    if compacted and 0 <= source_count <= len(raw): 
        return compacted + raw[source_count:]
    return raw

def _with_working_state_tail(messages, overlay) -> list[BaseMessage]:
    """Inject L3 working state ephemerally for the model API call (never persisted to state).

    Invariant: the ``<system-reminder>`` tail is call-ephemeral only — it must
    never be written into ``AgentState.messages``. Fresh user turn (last message
    is human): append the reminder onto that message. Tool loop (last message is
    AI or tool): append a separate tail HumanMessage so the user's text stays
    byte-stable for prefix caching.
    """
    if not overlay.strip():
        return list(messages)
    reminder = wrap_system_reminder(overlay)
    if not reminder:
        return list(messages)
    block = f"\n\n{reminder}"
    result = list(messages)
    if result and result[-1].type == "human":
        last = result[-1]
        if isinstance(last.content, str):
            result[-1] = HumanMessage(content=last.content + block)
            return result
        if isinstance(last.content, list):
            result[-1] = HumanMessage(
                content=[*last.content, {"type": "text", "text": block.lstrip()}]
            )
            return result
    return result + [HumanMessage(content=reminder)]

def _needs_approval(name, args, options, permission_store, tools_reg: ToolRegistry) -> bool:
    """Decide whether to ask the user for approval before running a tool."""
    if not options.enable_approval: 
        return False

    d = permission_store.check(name, args)
    if d in ("allow", "deny"): 
        return False
    
    # check if the tool is destructive
    return tools_reg.is_destructive(name, args)

def _denied_messages(calls, content) -> AgentState:
    """Build messages to inject when tool execution is denied by approval gate."""
    return {
        "messages": [
            ToolMessage(tool_call_id=call_id, name=name, content=content)
            for name, _, call_id in calls
        ],
        "approval_declined": True,
    }

def _reflection_token_delta(messages, since_index) -> int:
    """Estimate tokens in messages not yet covered by the last reflection run."""
    rec = list(messages)[max(0, since_index):]
    if not rec: 
        return 0
    return resolve_token_count(rec, known_input_tokens=None)

def _result_status(result) -> str | None:
    for line in result.splitlines()[:8]:
        if line.startswith("status="): 
            # remove the status= prefix and strip the whitespace
            return line.removeprefix("status=").strip() or None
    return None

def _tool_event(name, args, result, duration, *, call_id="", exit_status=None) -> dict:
    if exit_status is None:
        exit_status = _result_status(result) or (
            "error" if result.startswith("Error:") or result.startswith("Hook veto:") else "ok"
        )

    return {
        "kind": "tool",
        "tool": name,
        "args": args,
        "result": result,
        "call_id": call_id,
        "duration_ms": duration,
        "exit": exit_status,
    }

def extract_tool_calls(msg: AIMessage) -> list[tuple[str, dict[str, Any], str]]:
    out = []
    for idx, tc in enumerate(msg.tool_calls or []):
        out.append((tc.get("name", "unknown"), tc.get("args", {}), tc.get("id") or f"native-{idx}"))
    return out