from __future__ import annotations
from pathlib import Path
import difflib
from liteharness.tools import TOOL_MAP
from langchain_core.tools import BaseTool
from typing import Any

def normalize_tool(item: Any) -> BaseTool:
    """Normalize a user-supplied tool specifier to a ``BaseTool`` instance.

    Accepted forms:
    - ``BaseTool`` instances are returned unchanged.
    - Plain callables (functions, lambdas) are wrapped with
      :class:`langchain_core.tools.StructuredTool.from_function` so SDK
      users can pass `def my_tool(...) -> str` directly.
    - Strings are resolved against the SDK's built-in tool map by name,
      allowing users to mix-and-match built-ins with custom tools,
      e.g. ``tools=["read", "grep", my_custom_tool]``.
    """
    if isinstance(item, BaseTool):
        return item
    if callable(item):
        from langchain_core.tools import StructuredTool
        return StructuredTool.from_function(item)
    if isinstance(item, str):
        name = item.strip()
        t = TOOL_MAP.get(name)
        if t is None:
            raise ValueError(
                f"unknown tool name {name!r}; pass a BaseTool, a callable, "
                f"or one of: {', '.join(sorted(TOOL_MAP))}"
            )
        return t
    raise TypeError(
        f"unsupported tool type {type(item).__name__}; expected BaseTool, "
        f"callable, or tool-name string"
    )

def preview_diff(tool: str, args: dict) -> str:
    """Preview a proposed file edit without writing it."""
    path = args.get("path", "")
    if not path:
        return f"{tool}({args})"

    p = Path(path)
    try:
        old = p.read_text(encoding="utf-8") if p.exists() else ""
    except (OSError, UnicodeDecodeError) as exc:
        return f"Cannot read {path}: {exc}"

    if tool == "write":
        new = str(args.get("content", ""))
    elif tool == "edit":
        new = old
        for edit in args.get("edits", []):
            old_s = str(edit.get("old_string", ""))
            new_s = str(edit.get("new_string", ""))
            count = -1 if edit.get("replace_all") else 1
            new = new.replace(old_s, new_s, count)
    else:
        return f"{tool}({args})"

    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
        )
    )