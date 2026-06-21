from typing import Any

from langchain_core.messages import AIMessage


def extract_tool_calls(msg: AIMessage) -> list[tuple[str, dict[str, Any], str]]:
    """Return native tool calls as (name, args, call_id)."""
    out: list[tuple[str, dict[str, Any], str]] = []
    for idx, tc in enumerate(msg.tool_calls or []):
        name = tc.get("name", "unknown")
        args = tc.get("args", {})
        out.append((name, args, tc.get("id") or f"native-{idx}"))
    return out
