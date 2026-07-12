from __future__ import annotations

import asyncio
import re
import time
import uuid
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from liteharness.session_context import get_session_context

DEFAULT_AGENT_TOOLS = ("read", "grep", "glob", "web_search", "webfetch", "skill_view")
MAX_BATCH_TASKS = 8
DEFAULT_MAX_CONCURRENCY = 3 # max concurrent subagents to run
MAX_CONCURRENCY = 8
DEFAULT_TIMEOUT_SECONDS = 300 # timeout for a single subagent
MAX_TIMEOUT_SECONDS = 1_800

# Syntactic guard before names are used in filesystem paths (blocks traversal chars).
AGENT_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")

DEFAULT_SUBAGENT_TEMPLATE = """You are the {agent_name} LiteHarness subagent.

You were spawned by a parent agent to handle one scoped assignment. You run in an isolated session with no access to the parent conversation. Your final assistant message is returned directly to the parent—there is no follow-up turn with you.

Parent context:
{parent_context}

Role-specific instructions:
{agent_body}

Operating constraints:
- Read-only only. Use only the tools bound to this run.
- Stay within the parent request above.
- Prefer targeted investigation.
- Return one concise final message for the parent agent.
"""

# ContextVars for subagent specific memory pockets to avoid collisions between parallel subagent runs
_subagent_model: ContextVar[Any | None] = ContextVar("subagent_model", default=None)
_parent_thread_id: ContextVar[str | None] = ContextVar("parent_thread_id", default=None)
_active_subagent_runs = 0

# Cross-call concurrency cap shared by every spawn_subagent invocation in this
# process. A single spawn_subagent call already bounds in-call concurrency via an
# asyncio.Semaphore, but without this global gate the main agent could fire
# multiple spawn_subagent calls in one turn and exceed MAX_CONCURRENCY. Lazily
# created on first use so we bind to the running event loop.
_global_concurrency_semaphore: asyncio.Semaphore | None = None


def _agents_dir() -> Path:
    from liteharness.session_context import try_get_session_context

    rt = try_get_session_context()
    if rt is not None:
        return rt.ness_dir / "agents"
    return Path(".ness") / "agents"


def _build_subagent_prompt(agent_name: str, agent_body: str, parent_context: str = "") -> str:
    rt = get_session_context()
    template = DEFAULT_SUBAGENT_TEMPLATE
    cfg = rt.agent_config
    if cfg is not None and cfg.task_prompts.subagent:
        src = cfg.task_prompts.subagent
        if callable(src):
            template = src()
        else:
            text = str(src)
            if "\n" not in text and len(text) < 512:
                p = Path(text)
                try:
                    if p.exists() and p.is_file():
                        template = p.read_text(encoding="utf-8").strip()
                    else:
                        template = text
                except OSError:
                    template = text
            else:
                template = text
    elif cfg is not None and cfg.subagents and cfg.subagents.prompt_template:
        template = cfg.subagents.prompt_template
    return template.format(
        agent_name=agent_name,
        agent_body=agent_body.strip(),
        parent_context=parent_context.strip(),
    )


def _global_semaphore() -> asyncio.Semaphore:
    global _global_concurrency_semaphore
    if _global_concurrency_semaphore is None:
        _global_concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    return _global_concurrency_semaphore


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
    agents_dir = _agents_dir()
    if not agents_dir.is_dir():
        return frozenset()
    return frozenset(
        path.stem for path in agents_dir.glob("*.md") if path.is_file() and path.stem
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
    """Run one or more isolated sub-agents for parallel investigation and wait for results."""
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

    # In-call semaphore bounds concurrent tasks inside this invocation; the
    # global semaphore bounds total subagents across concurrent spawn_subagent
    # calls in the same process.
    in_call_semaphore = asyncio.Semaphore(min(concurrency, len(validated)))

    # run one subagent
    async def run_one(index: int, task: SubagentTask) -> SubagentRunResult:
        # in-call slot first so a failed prep doesn't waste the global slot
        async with in_call_semaphore:
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
            # global slot can be released as soon as the subagent finishes
            async with _global_semaphore():
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
    agent_prompt = _build_subagent_prompt(task.name, meta.get("prompt", ""), parent_context)
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
        get_session_context().thread_store.register_subagent(
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
        get_session_context().thread_store.complete_subagent(
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
    """Main subagent invocation using the SDK graph builder."""
    global _active_subagent_runs
    _active_subagent_runs += 1
    try:
        from liteharness.graph.builder import build_graph
        from liteharness.tools import ToolRegistry

        rt = get_session_context()
        parent = rt.agent_config
        if parent is None:
            raise RuntimeError("Subagent requires agent_config on SessionContext")

        child_options = replace(
            parent.options,
            enable_approval=False,
            session_end_reflection=False,
            reflection_token_ratio=0.0,
        )
        child_cfg = replace(
            parent,
            model=model or parent.model,
            tools=list(prepared.tools),
            tool_registry=ToolRegistry(prepared.tools),
            overlay=None,
            modes=None,
            approval_handler=None,
            question_handler=None,
            options=child_options,
            cost_tracker=parent.cost_tracker,
            tracer=parent.tracer,
            memory_store=parent.memory_store,
            thread_store=parent.thread_store,
            permission_store=parent.permission_store,
            hook_runner=parent.hook_runner,
            skill_loader=parent.skill_loader,
        )
        app = build_graph(
            child_cfg,
            thread_id=thread_id,
            agent_mode="act",
            git_available=False,
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
        # Only the truly final message is the answer. Returning an earlier AI
        # message when the run ended on a tool/human message would silently
        # surface stale output (e.g. after an interrupt or tool error).
        if not messages:
            return "Subagent completed without producing any messages"
        last = messages[-1]
        if getattr(last, "type", "") in ("ai", "assistant"):
            final = last.content
        else:
            return (
                "Subagent did not finish with an assistant message "
                f"(last message type: {getattr(last, 'type', 'unknown')})"
            )
        return str(final or "Subagent completed with an empty final message")
    finally:
        _active_subagent_runs -= 1


def _load_agent(name: str) -> dict[str, Any] | str:
    # load the agent specific instructions from the .ness/agents folder and put them in the prompt
    path = _agents_dir() / f"{name}.md"
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
    from liteharness.tools import READ_ONLY_TOOLS, TOOL_MAP

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
