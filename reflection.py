from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from config import settings
from memory import MAX_NESS_CHARS, append_memory, memory_char_count, memory_key
from parsers import extract_tool_calls
from context import build_reflection_prompt
from tools.fs import edit_file, read_file

MAX_REFLECTION_TOOL_CALLS = 3
MEMORY_REL_PATH = str(Path(settings.ness_dir) / "NESS.md")


@dataclass(frozen=True)
class ReflectionResult:
    durable_changed: bool = False
    error: str = ""
    over_limit: bool = False


def _size_header() -> str:
    current = memory_char_count()
    status = "OK" if current <= MAX_NESS_CHARS else "OVER_LIMIT"
    return f"NESS.md size: {current}/{MAX_NESS_CHARS} chars ({status})\n\n"


def _limit_message() -> str:
    return (
        f"NESS.md is {memory_char_count()} chars (limit {MAX_NESS_CHARS}). "
        "Use edit_memory to compress or remove stale content."
    )


@tool
def read_memory() -> str:
    """Read .ness/NESS.md with a size header."""
    return _size_header() + read_file.func(path=MEMORY_REL_PATH)


@tool
def add_to_memory(text: str) -> str:
    """Append durable notes to .ness/NESS.md."""
    current = memory_char_count()
    addition = text.strip() + "\n"
    if current + len(addition) > MAX_NESS_CHARS:
        return (
            f"Error: append would exceed {MAX_NESS_CHARS} char limit "
            f"({current} + {len(addition)} > {MAX_NESS_CHARS}). "
            f"{_limit_message()}"
        )
    result = append_memory(addition)
    if memory_char_count() > MAX_NESS_CHARS:
        return f"{result}\nWarning: {_limit_message()}"
    return result


@tool
def edit_memory(old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Edit .ness/NESS.md to update conventions or compress content."""
    result = edit_file.func(
        path=MEMORY_REL_PATH,
        old_string=old_string,
        new_string=new_string,
        replace_all=replace_all,
    )
    if result.startswith("Error"):
        return result
    if memory_char_count() > MAX_NESS_CHARS:
        return f"{result}\nWarning: {_limit_message()}"
    return result


REFLECTION_TOOLS = [read_memory, add_to_memory, edit_memory]
REFLECTION_TOOL_NAMES = [t.name for t in REFLECTION_TOOLS]
REFLECTION_TOOL_MAP = {t.name: t for t in REFLECTION_TOOLS}


async def run_reflection_gate(
    thread_id: str,
    messages: Iterable[BaseMessage],
    model,
    user_message_count: int,
) -> ReflectionResult:
    """Run a bounded tool loop: read -> write -> optional compress."""
    if model is None:
        return ReflectionResult()

    # get the before memory key and build the reflection prompt
    before_key = memory_key()
    prompt = build_reflection_prompt(
        thread_id,
        messages,
        user_message_count,
        max_tool_calls=MAX_REFLECTION_TOOL_CALLS,
        max_ness_chars=MAX_NESS_CHARS,
    )

    try:
        tool_model = model.bind_tools(REFLECTION_TOOLS)
        transcript: list[BaseMessage] = [HumanMessage(content=prompt)]
        tool_calls_used = 0

        # run the tool loop upto max tool calls
        while tool_calls_used < MAX_REFLECTION_TOOL_CALLS:
            response: AIMessage = await tool_model.ainvoke(transcript)
            calls = extract_tool_calls(response, REFLECTION_TOOL_NAMES, settings.mode)
            if not calls:
                break

            # limit the number of tool calls to the remaining number of tool calls
            remaining = MAX_REFLECTION_TOOL_CALLS - tool_calls_used
            calls = calls[:remaining]
            tool_calls_used += len(calls)

            # append the response to the transcript
            transcript.append(response)

            # iterate over the tool calls and append the tool messages to the transcript
            for name, args, call_id in calls:
                tool = REFLECTION_TOOL_MAP.get(name)
                try:
                    content = (
                        str(tool.invoke(args))
                        if tool
                        else f"Error: unknown reflection tool {name}"
                    )
                except Exception as exc:
                    content = f"Error: {exc}"
                transcript.append(
                    ToolMessage(
                        tool_call_id=call_id or f"reflect-{uuid.uuid4().hex[:8]}",
                        name=name,
                        content=content,
                    )
                )

            if tool_calls_used >= MAX_REFLECTION_TOOL_CALLS:
                break

            # if the memory is over the chars limit, append a human message to the transcript
            if memory_char_count() > MAX_NESS_CHARS:
                transcript.append(
                    HumanMessage(
                        content=(
                            f"System: {_limit_message()} "
                            f"You have {MAX_REFLECTION_TOOL_CALLS - tool_calls_used} tool call(s) left."
                        )
                    )
                )

    except Exception as exc:
        return ReflectionResult(error=str(exc))

    after_count = memory_char_count()
    return ReflectionResult(
        durable_changed=memory_key() != before_key, # check against the last modified time and size
        over_limit=after_count > MAX_NESS_CHARS,
    )
