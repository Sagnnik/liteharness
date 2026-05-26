import re
import json
from langchain_core.messages import AIMessage

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that models often wrap XML in."""
    text = re.sub(r"```xml\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text

def parse_xml_tools(text: str, tool_names: list[str] | None = None):
    """Parse XML tool calls from model output. Tool names defaults to dynamic lists from registry.
    Also I am figuring out re as well. So just a cheatsheet for me.
    . - Any character (except newline)
    | - Either or (pipe)
    () - Grouping
    [] - Character class
    {} - Quantifier
    <name> - Named group
    ? - Optional
    * - 0 or more
    + - 1 or more
    ^ - Start of string, [^] - Not start of string
    $ - End of string, [$] - Not end of string
    \ - Escape character
    \d - Digit,  \D - Not digit
    \w - Word character, \W - Not word character
    \s - Whitespace, \S - Not whitespace
    \b - Word boundary, \B - Not word boundary
    """

    if not text:
        return []

    text = strip_markdown_fences(text)
    
    names = "|".join(re.escape(n) for n in tool_names) if tool_names else r"\w+"
    pattern = rf"<({names})>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    calls = []
    param_pattern = r"<(\w+)>(.*?)</\1>"

    for name, body in matches:
        params = {k: v.strip() for k, v in re.findall(param_pattern, body, re.DOTALL)}
        calls.append((name, params))
    return calls

def extract_tool_calls(msg: AIMessage, tool_names: list[str], mode: str):
    """Unified extractor: Native is JSON mode or XML fallback"""
    if mode == "json" and getattr(msg, "tool_calls", None):
        out = []
        for tc in msg.tool_calls:
            name = tc.get("name", "unknown")
            args = tc.get("args", {})
            if not isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    args = {}

            out.append((name, args, tc.get("id", "")))

        return out

    xml = parse_xml_tools(msg.content or "", tool_names)
    return [(n, a, f"xml-{i}") for i, (n, a) in enumerate(xml)]

def format_tool_result(name: str, result: str) -> str:
    """Full result — NO truncation from before."""
    return f"[{name}] => {result}"