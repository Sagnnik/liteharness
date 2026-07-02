from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from config import settings
from context import build_subagent_prompt
from session import complete_subagent, register_subagent

# gets the subagent instructions
AGENTS_DIR = Path(settings.ness_dir) / "agents"

DEFAULT_AGENT_TOOLS = ("read_file", "grep", "glob_files", "list_files")
MAX_BATCH_TASKS = 8
DEFAULT_MAX_CONCURRENCY = 3 # max concurrent subagents to run
MAX_CONCURRENCY = 8
DEFAULT_TIMEOUT_SECONDS = 300 # timeout for a single subagent
MAX_TIMEOUT_SECONDS = 1_800

# Syntactic guard before names are used in filesystem paths (blocks traversal chars).
AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

# ContextVars for subagent specific memory pockets to avoid collisions
_subagent_model: ContextVar[Any | None] = ContextVar("subagent_model", default=None)
_parent_thread_id: ContextVar[str | None] = ContextVar("parent_thread_id", default=None)
_active_subagent_runs = 0


@dataclass(frozen=True)
class SubagentTask:
    name: str
    prompt: str
    label: str = ""


@dataclass(frozen=True)
class PreparedTask:
    task: SubagentTask
    agent_prompt: str
    tools: list[Any]


@dataclass(frozen=True)
class SubagentRunResult:
    index: int
    name: str
    label: str
    status: str
    duration_ms: int
    thread_id: str
    output: str

# sets the subagent runtime context before main agent call; setter doesn't need to be async
def set_subagent_runtime(model: Any | None, parent_thread_id: str | None = None) -> None:
    _subagent_model.set(model)
    if parent_thread_id is not None:
        _parent_thread_id.set(parent_thread_id)


def subagent_runs_active() -> int:
    """Number of subagent graphs currently executing (nested CLI events should be hidden)."""
    return _active_subagent_runs

def _available_agent_names() -> frozenset[str]:
    if not AGENTS_DIR.is_dir():
        return frozenset()
    return frozenset(
        path.stem for path in AGENTS_DIR.glob("*.md") if path.is_file() and path.stem
    )


def _spawn_subagent_description() -> str:
    agents = ", ".join(sorted(_available_agent_names())) or "none"
    return f"""Run one or more read-only LiteHarness subagents and wait for all results.

Use exactly one invocation pattern:
1. Single task: pass name and prompt.
   Example: spawn_subagent(name="explore", prompt="Find route handlers")
2. Batch: pass tasks only (do not also pass name or prompt).
   Example: spawn_subagent(tasks=[
     {{"name": "explore", "prompt": "Find routes", "label": "routes"}},
     {{"name": "explore", "prompt": "Find tests", "label": "tests"}},
   ])

Each task object requires name and prompt; label is optional.
max_concurrency defaults to {DEFAULT_MAX_CONCURRENCY}; timeout defaults to {DEFAULT_TIMEOUT_SECONDS}s.

Agent names must match a file in .ness/agents/<name>.md.
Available agents: {agents}."""


@tool(description=_spawn_subagent_description())
async def spawn_subagent(
    name: str = "",
    prompt: str = "",
    tasks: list[dict[str, str]] | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run one or more read-only LiteHarness subagents and wait for all results."""
    started = time.time()
    batch_invocation = tasks is not None
    # check the runtime context
    runtime = _check_runtime()
    if isinstance(runtime, str):
        return _call_error(runtime, batch_invocation, started, 0)

    model = runtime

    # validate the spawn request
    validated = _validate_spawn_request(name, prompt, tasks)
    if isinstance(validated, str):
        return _call_error(validated, batch_invocation, started, 0)

    # validate the max concurrency
    concurrency = _bounded_int(max_concurrency, "max_concurrency", 1, MAX_CONCURRENCY)
    if isinstance(concurrency, str):
        return _call_error(concurrency, batch_invocation, started, len(validated))

    # validate the timeout
    timeout_seconds = _bounded_int(timeout, "timeout", 1, MAX_TIMEOUT_SECONDS)
    if isinstance(timeout_seconds, str):
        return _call_error(timeout_seconds, batch_invocation, started, len(validated))

    # semaphore to limit the number of concurrent subagents
    semaphore = asyncio.Semaphore(min(concurrency, len(validated)))

    # run one subagent
    async def run_one(index: int, task: SubagentTask) -> SubagentRunResult:
        # wait for the semaphore to be available
        async with semaphore:
            prepared = _prepare_task(task)
            if isinstance(prepared, str):
                return SubagentRunResult(
                    index=index,
                    name=task.name,
                    label=task.label,
                    status="failed",
                    duration_ms=0,
                    thread_id=f"subagent-{task.name}-prep-failed",
                    output=prepared,
                )
            # run the subagent with a timeout
            return await _run_prepared_with_timeout(
                index=index,
                prepared=prepared,
                model=model,
                timeout_seconds=timeout_seconds,
            )

    # run all the subagents
    # we can await all the subagents at once here; subagents should block main agent run
    results = await asyncio.gather(
        *(run_one(index, task) for index, task in enumerate(validated, start=1))
    )
    # format the result
    if not batch_invocation:
        return _format_single_result(results[0])
    return _format_batch_result(results, int((time.time() - started) * 1000))


def _tool_error(message: str) -> str:
    return f"Error: {message}"


def _call_error(message: str, batch: bool, started: float, task_count: int) -> str:
    duration_ms = int((time.time() - started) * 1000)
    if not batch:
        return f"{_tool_error(message)} (duration_ms={duration_ms})"
    return _format_batch_error(message, duration_ms, task_count)


def _check_runtime() -> Any | str:
    model = _subagent_model.get()
    if model is None:
        return "no model available for subagent"
    return model


def _validate_spawn_request(
    name: str,
    prompt: str,
    tasks: list[dict[str, str]] | None,
) -> list[SubagentTask] | str:
    if tasks is None:
        normalized = _normalize_task({"name": name, "prompt": prompt}, 1)
        if isinstance(normalized, str):
            return normalized
        return [normalized]

    if str(name or "").strip() or str(prompt or "").strip():
        return "provide either tasks or name/prompt, not both"

    normalized = _normalize_tasks(tasks)
    if isinstance(normalized, str):
        return normalized

    return normalized


def _normalize_tasks(raw_tasks: list[dict[str, str]]) -> list[SubagentTask] | str:
    if not isinstance(raw_tasks, list):
        return "tasks must be a list of task objects"
    if not raw_tasks:
        return "spawn_subagent requires at least one task"
    if len(raw_tasks) > MAX_BATCH_TASKS:
        return f"spawn_subagent supports at most {MAX_BATCH_TASKS} tasks"

    normalized: list[SubagentTask] = []
    for index, task in enumerate(raw_tasks, start=1):
        item = _normalize_task(task, index)
        if isinstance(item, str):
            return item
        normalized.append(item)
    return normalized


def _normalize_task(raw_task: Any, index: int) -> SubagentTask | str:
    if not isinstance(raw_task, dict):
        return f"task {index} must be an object with name and prompt"
    name = _string_field(raw_task, "name").strip()
    prompt = _string_field(raw_task, "prompt").strip()
    label = _string_field(raw_task, "label").strip()
    if error := _validate_agent_name(name):
        return error
    if not prompt:
        return f"task {index} prompt cannot be empty"
    return SubagentTask(name=name, prompt=prompt, label=label)


def _string_field(raw_task: dict[str, Any], key: str) -> str:
    value = raw_task.get(key, "")
    return "" if value is None else str(value)


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int | str:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return f"{name} must be an integer"
    if parsed < minimum:
        return f"{name} must be at least {minimum}"
    return min(parsed, maximum)


def _prepare_task(task: SubagentTask) -> PreparedTask | str:
    meta = _load_agent(task.name)
    if isinstance(meta, str):
        return meta

    tool_names = _agent_tool_names(meta, task.name)
    if isinstance(tool_names, str):
        return tool_names

    tools = _resolve_tools(task.name, tool_names)
    if isinstance(tools, str):
        return tools

    parent_context = _parent_context(task)
    agent_prompt = build_subagent_prompt(task.name, meta.get("prompt", ""), parent_context)
    return PreparedTask(task=task, agent_prompt=agent_prompt, tools=tools)


def _parent_context(task: SubagentTask) -> str:
    lines = []
    if task.label:
        lines.append(f"Task label: {task.label}")
    lines.append(f"Parent request: {task.prompt}")
    return "\n".join(lines)


async def _run_prepared_with_timeout(
    index: int,
    prepared: PreparedTask,
    model: Any,
    timeout_seconds: int,
) -> SubagentRunResult:
    task = prepared.task
    started = time.time()
    thread_id = f"subagent-{task.name}-{uuid.uuid4().hex[:8]}"
    parent_thread_id = _parent_thread_id.get()

    # register the subagent with the parent thread in threads db
    if parent_thread_id:
        register_subagent(
            parent_thread_id,
            thread_id,
            agent_name=task.name,
            label=task.label,
        )

    try:
        # .wait_for(): wrap a single coroutine with timeout and handle exceptions
        output = await asyncio.wait_for(
            _invoke_subagent(
                prepared=prepared,
                model=model,
                thread_id=thread_id,
            ),
            timeout=timeout_seconds,
        )
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="ok",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=output,
        )
    # handle exceptions
    except TimeoutError:
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="timeout",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=f"timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        message = str(exc)
        if message.startswith("Error: "):
            message = message[7:]
        result = SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="failed",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=message,
        )

    # complete the subagent run with the result stored in the threads db
    if parent_thread_id:
        complete_subagent(
            thread_id,
            status=result.status,
            output=result.output,
            duration_ms=result.duration_ms,
        )
    return result


async def _invoke_subagent(
    prepared: PreparedTask,
    model: Any,
    thread_id: str,
) -> str:
    """Main subagent invocation; uses the build_graph from agent.py"""
    global _active_subagent_runs
    _active_subagent_runs += 1
    try:
        from agent import build_graph

        app = build_graph(
            model,
            tools=prepared.tools, # read-only tools
            thread_id=thread_id,
            agent_mode="act", # subagents dont have plan mode
        )
        result = await app.ainvoke(
            {
                "messages": [HumanMessage(content=prepared.agent_prompt)],
                "approval_declined": False,
                "todos": [],
                "agent_mode": "act",
            },
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        final = next((m.content for m in reversed(messages) if m.type in ("ai", "assistant")), "")
        # return the final assistant message or a default message if no final message
        return str(final or "Subagent completed without a final message")
    finally:
        _active_subagent_runs -= 1


def _load_agent(name: str) -> dict[str, Any] | str:
    # load the agent specific instructions from the .ness/agents folder and put them in the prompt
    path = AGENTS_DIR / f"{name}.md"
    if not path.exists():
        return f"No agent definition at {path}"

    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            if not isinstance(meta, dict):
                meta = {}
            meta["prompt"] = parts[2].strip()
            return meta
    return {"prompt": text.strip(), "tools": list(DEFAULT_AGENT_TOOLS)}


def _validate_agent_name(name: str) -> str | None:
    if not name or not AGENT_NAME_RE.fullmatch(name):
        return (
            "agent name must start with a letter or number and contain only letters, "
            "numbers, dots, underscores, or hyphens"
        )
    available = _available_agent_names()
    if name not in available:
        listed = ", ".join(sorted(available)) or "none"
        return f"unknown agent '{name}'; available agents: {listed}"
    return None


def _agent_tool_names(meta: dict[str, Any], agent_name: str) -> set[str] | str:
    raw_tools = meta.get("tools", list(DEFAULT_AGENT_TOOLS))
    if not all(isinstance(item, str) for item in raw_tools):
        return f"tools for subagent {agent_name} must be a list of tool names"
    tool_names = {item.strip() for item in raw_tools if item.strip()}
    if not tool_names:
        return f"subagent {agent_name} has no configured tools"
    return tool_names


def _resolve_tools(agent_name: str, tool_names: set[str]) -> list[Any] | str:
    # TODO: In future if subagents need more tool freedom; change access here
    # resolve tools from string to tool objects
    from tools import READ_ONLY_TOOLS, TOOL_MAP

    known_tool_names = set(TOOL_MAP)
    # reject unknown tools
    unknown = sorted(tool_names - known_tool_names)
    if unknown:
        return f"unknown tools for subagent {agent_name}: {', '.join(unknown)}"

    # reject unsafe tools
    subagent_safe_tools = set(READ_ONLY_TOOLS) - {
        "spawn_subagent",
        "todo",
        "search_tools",
        "add_tools",
    }
    unsafe = sorted(tool_names - subagent_safe_tools)
    if unsafe:
        return (
            "spawn_subagent only allows read-only native tools; "
            f"unsafe tools for subagent {agent_name}: {', '.join(unsafe)}"
        )

    return [TOOL_MAP[name] for name in sorted(tool_names)]


def _format_single_result(result: SubagentRunResult) -> str:
    if result.status == "ok":
        return result.output
    return f"Error: subagent {result.name} {result.status}: {result.output}"


def _format_batch_result(results: list[SubagentRunResult], duration_ms: int) -> str:
    ok = sum(1 for result in results if result.status == "ok")
    failed = len(results) - ok
    status = "ok" if failed == 0 else "failed" if ok == 0 else "partial"
    lines = [
        f"status={status}",
        f"duration_ms={duration_ms}",
        f"tasks_total={len(results)}",
        f"tasks_ok={ok}",
        f"tasks_failed={failed}",
        "",
    ]
    for result in results:
        heading = (
            f"[{result.index}] name={result.name} status={result.status} "
            f"duration_ms={result.duration_ms} thread_id={result.thread_id}"
        )
        if result.label:
            heading += f" label={result.label}"
        output = result.output.strip()
        if result.status != "ok":
            output = f"error={output}"
        lines.extend([heading, output, ""])
    return "\n".join(lines).rstrip()


def _format_batch_error(error: str, duration_ms: int, task_count: int = 0) -> str:
    return "\n".join(
        [
            "status=error",
            f"duration_ms={duration_ms}",
            f"tasks_total={task_count}",
            "tasks_ok=0",
            f"tasks_failed={task_count}",
            f"error={error}",
        ]
    )
