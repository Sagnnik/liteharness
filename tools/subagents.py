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

AGENTS_DIR = Path(settings.ness_dir) / "agents"

DEFAULT_AGENT_TOOLS = ("read_file", "grep", "glob_files", "list_files")
MAX_DEPTH = 2
MAX_BATCH_TASKS = 8
DEFAULT_MAX_CONCURRENCY = 3
MAX_CONCURRENCY = 8
DEFAULT_TIMEOUT_SECONDS = 300
MAX_TIMEOUT_SECONDS = 1_800

AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

_subagent_model: ContextVar[Any | None] = ContextVar("subagent_model", default=None)
_subagent_depth: ContextVar[int] = ContextVar("subagent_depth", default=0)


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


def set_subagent_runtime(model: Any | None, depth: int) -> None:
    _subagent_model.set(model)
    _subagent_depth.set(depth)


@tool
async def spawn_subagent(
    name: str = "",
    prompt: str = "",
    tasks: list[dict[str, str]] | None = None,
    num_subagents: int | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Run one or more read-only LiteHarness subagents and wait for all results."""
    started = time.time()
    task_count = 0
    batch_invocation = tasks is not None
    try:
        depth, model = _runtime()
        normalized = _normalize_spawn_request(name, prompt, tasks, num_subagents)
        task_count = len(normalized)
        concurrency = _bounded_int(max_concurrency, "max_concurrency", 1, MAX_CONCURRENCY)
        timeout_seconds = _bounded_int(timeout, "timeout", 1, MAX_TIMEOUT_SECONDS)

        prepared = [_prepare_task(task) for task in normalized]
        semaphore = asyncio.Semaphore(min(concurrency, len(prepared)))

        async def run_one(index: int, item: PreparedTask) -> SubagentRunResult:
            async with semaphore:
                return await _run_prepared_with_timeout(
                    index=index,
                    prepared=item,
                    model=model,
                    depth=depth,
                    timeout_seconds=timeout_seconds,
                )

        results = await asyncio.gather(
            *(run_one(index, item) for index, item in enumerate(prepared, start=1))
        )
        if not batch_invocation:
            return _format_single_result(results[0])
        return _format_batch_result(results, int((time.time() - started) * 1000))
    except Exception as exc:
        if not batch_invocation:
            duration_ms = int((time.time() - started) * 1000)
            return f"Error: {exc} (duration_ms={duration_ms})"
        return _format_batch_error(str(exc), int((time.time() - started) * 1000), task_count)


def _runtime() -> tuple[int, Any]:
    depth = _subagent_depth.get()
    if depth >= MAX_DEPTH:
        raise RuntimeError("subagent depth limit reached")
    model = _subagent_model.get()
    if model is None:
        raise RuntimeError("no model available for subagent")
    return depth, model


def _normalize_spawn_request(
    name: str,
    prompt: str,
    tasks: list[dict[str, str]] | None,
    num_subagents: int | None,
) -> list[SubagentTask]:
    if tasks is None:
        normalized = [_normalize_task({"name": name, "prompt": prompt}, 1)]
    else:
        if str(name or "").strip() or str(prompt or "").strip():
            raise ValueError("provide either tasks or name/prompt, not both")
        normalized = _normalize_tasks(tasks)

    if num_subagents is not None:
        expected = _bounded_int(num_subagents, "num_subagents", 1, MAX_BATCH_TASKS)
        if expected != len(normalized):
            raise ValueError(
                f"num_subagents must equal task count ({len(normalized)}), got {expected}"
            )
    return normalized


def _normalize_tasks(raw_tasks: list[dict[str, str]]) -> list[SubagentTask]:
    if not isinstance(raw_tasks, list):
        raise ValueError("tasks must be a list of task objects")
    if not raw_tasks:
        raise ValueError("spawn_subagent requires at least one task")
    if len(raw_tasks) > MAX_BATCH_TASKS:
        raise ValueError(f"spawn_subagent supports at most {MAX_BATCH_TASKS} tasks")
    return [_normalize_task(task, index) for index, task in enumerate(raw_tasks, start=1)]


def _normalize_task(raw_task: Any, index: int) -> SubagentTask:
    if not isinstance(raw_task, dict):
        raise ValueError(f"task {index} must be an object with name and prompt")
    name = _string_field(raw_task, "name").strip()
    prompt = _string_field(raw_task, "prompt").strip()
    label = _string_field(raw_task, "label").strip()
    _validate_agent_name(name)
    if not prompt:
        raise ValueError(f"task {index} prompt cannot be empty")
    return SubagentTask(name=name, prompt=prompt, label=label)


def _string_field(raw_task: dict[str, Any], key: str) -> str:
    value = raw_task.get(key, "")
    return "" if value is None else str(value)


def _bounded_int(value: int, name: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if parsed < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    return min(parsed, maximum)


def _prepare_task(task: SubagentTask) -> PreparedTask:
    meta = _load_agent(task.name)
    tool_names = _agent_tool_names(meta, task.name)
    tools = _resolve_tools(task.name, tool_names)
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
    depth: int,
    timeout_seconds: int,
) -> SubagentRunResult:
    task = prepared.task
    started = time.time()
    thread_id = f"subagent-{task.name}-{uuid.uuid4().hex[:8]}"
    try:
        output = await asyncio.wait_for(
            _invoke_subagent(
                prepared=prepared,
                model=model,
                depth=depth,
                thread_id=thread_id,
            ),
            timeout=timeout_seconds,
        )
        return SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="ok",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=output,
        )
    except TimeoutError:
        return SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="timeout",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=f"Timed out after {timeout_seconds}s",
        )
    except Exception as exc:
        return SubagentRunResult(
            index=index,
            name=task.name,
            label=task.label,
            status="failed",
            duration_ms=int((time.time() - started) * 1000),
            thread_id=thread_id,
            output=str(exc),
        )


async def _invoke_subagent(
    prepared: PreparedTask,
    model: Any,
    depth: int,
    thread_id: str,
) -> str:
    from agent import build_graph

    app = build_graph(
        model,
        tools=prepared.tools,
        thread_id=thread_id,
        agent_mode="normal",
    )
    result = await app.ainvoke(
        {
            "messages": [HumanMessage(content=prepared.agent_prompt)],
            "approval_declined": False,
            "todos": [],
            "subagent_depth": depth + 1,
            "agent_mode": "normal",
        },
        config={"configurable": {"thread_id": thread_id}},
    )
    messages = result.get("messages", [])
    final = next((m.content for m in reversed(messages) if m.type in ("ai", "assistant")), "")
    return str(final or "Subagent completed without a final message")


def _load_agent(name: str) -> dict[str, Any]:
    _validate_agent_name(name)
    path = AGENTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No agent definition at {path}")

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


def _validate_agent_name(name: str) -> None:
    if not name or not AGENT_NAME_RE.fullmatch(name):
        raise ValueError(
            "agent name must start with a letter or number and contain only letters, numbers, dots, underscores, or hyphens"
        )


def _agent_tool_names(meta: dict[str, Any], agent_name: str) -> set[str]:
    raw_tools = meta.get("tools", list(DEFAULT_AGENT_TOOLS))
    if raw_tools is None:
        raw_tools = list(DEFAULT_AGENT_TOOLS)
    if not isinstance(raw_tools, list) or not all(isinstance(item, str) for item in raw_tools):
        raise ValueError(f"tools for subagent {agent_name} must be a list of tool names")
    tool_names = {item.strip() for item in raw_tools if item.strip()}
    if not tool_names:
        raise ValueError(f"subagent {agent_name} has no configured tools")
    return tool_names


def _resolve_tools(agent_name: str, tool_names: set[str]) -> list[Any]:
    from tools import READ_ONLY_TOOLS, TOOL_MAP, get_tools_for_names

    known_tool_names = set(TOOL_MAP)
    unknown = sorted(tool_names - known_tool_names)
    if unknown:
        raise ValueError(f"unknown tools for subagent {agent_name}: {', '.join(unknown)}")

    subagent_safe_tools = set(READ_ONLY_TOOLS) - {"spawn_subagent", "todo_write"}
    unsafe = sorted(tool_names - subagent_safe_tools)
    if unsafe:
        raise ValueError(
            "spawn_subagent only allows read-only native tools; "
            f"unsafe tools for subagent {agent_name}: {', '.join(unsafe)}"
        )

    tools = get_tools_for_names(tool_names)
    if len(tools) != len(tool_names):
        resolved = {tool.name for tool in tools}
        missing = sorted(tool_names - resolved)
        raise ValueError(f"subagent {agent_name} has unresolved tools: {', '.join(missing)}")
    return tools


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
    for result in sorted(results, key=lambda item: item.index):
        heading = (
            f"[{result.index}] name={result.name} status={result.status} "
            f"duration_ms={result.duration_ms} thread_id={result.thread_id}"
        )
        if result.label:
            heading += f" label={result.label}"
        lines.extend([heading, result.output.strip(), ""])
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
