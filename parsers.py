import re
from typing import Any

from langchain_core.messages import AIMessage


def parse_xml_tools(text: str, tool_names: list[str]) -> list[tuple[str, dict[str, str]]]:
    """Parse XML fallback tool calls from model text."""
    text = re.sub(r"```xml\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    names = "|".join(re.escape(n) for n in tool_names)
    pattern = rf"<({names})>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    calls: list[tuple[str, dict[str, str]]] = []
    param_pattern = r"<(\w+)>(.*?)</\1>"
    for name, body in matches:
        params = {k: v.strip() for k, v in re.findall(param_pattern, body, re.DOTALL)}
        calls.append((name, params))
    return calls


def extract_tool_calls(msg: AIMessage, tool_names: list[str], mode: str) -> list[tuple[str, dict[str, Any], str]]:
    """Return tool calls as (name, args, call_id) for native or XML mode."""
    if mode == "json" and msg.tool_calls:
        out: list[tuple[str, dict[str, Any], str]] = []
        for idx, tc in enumerate(msg.tool_calls):
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            out.append((name, args, tc.get("id") or f"native-{idx}"))
        return out

    xml = parse_xml_tools(str(msg.content or ""), tool_names)
    return [(name, args, f"xml-{idx}") for idx, (name, args) in enumerate(xml)]
