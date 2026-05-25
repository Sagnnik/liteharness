import re

def strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences that models often wrap XML in."""
    text = re.sub(r"```xml\s*", "", text)
    text = re.sub(r"```\s*", "", text)
    return text

def parse_xml_tools(text: str):
    """Parse XML tool calls from model output.
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
    text = strip_markdown_fences(text)
    calls = []
    tool_names = "read_file|write_file|apply_diff|list_files|get_project_context|search_files|run_tests|git_snapshot|git_diff"
    pattern = rf"<({tool_names})>(.*?)</\1>"
    matches = re.findall(pattern, text, re.DOTALL)

    for name, body in matches:
        params = {}
        param_pattern = r"<(path|content|old_string|new_string|query|search|replace|message)>(.*?)</\1>"
        for p_name, p_val in re.findall(param_pattern, body, re.DOTALL):
            params[p_name] = p_val.strip()
        calls.append((name, params))

    return calls

"""
format tool result
eg. list_files:
index.html
styles.css
script.js
package.json
README.md

OUTPUT: [list_files] => index.html styles.css script.js package.json README.md
"""

def format_tool_result(name: str, result: str) -> str:
    preview = str(result).replace("\n", " ")[:250]
    return f"[{name}] => {preview}"