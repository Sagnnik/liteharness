from __future__ import annotations

import asyncio
import time
from typing import Annotated, Any, Awaitable, Callable, Literal, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from config import _usage_value, cost_tracker, settings
from model import create_compaction_model, create_reflection_model
from hooks import run_hooks
from memory import (
    load_ness_memory,
    load_repo_context,
    load_session_memory,
    load_user_memory,
    ness_key,
    user_memory_key,
)
from parsers import extract_tool_calls
from permissions import check_with_rule, default_rule_for, persist_rule
from context import (
    DEFAULT_PERSONA,
    DEFAULT_PERSONA_ID,
    build_l0,
    build_l1,
    build_project_context_block,
    build_working_state_overlay,
    render_todos,
)
from reflection import (
    consume_reflection_message_index,
    is_reflection_running,
    run_reflection_gate,
)
from session import append_event
from skill_loader import load_skills, select_sticky_skills
from tools import (
    ALL_TOOLS,
    is_destructive_tool_call,
    is_git_repo,
    is_read_only_tool_call,
    select_tools_for_session,
    tools_generation,
)
from tools.ask import QuestionHandler, set_question_runtime
from tools.git import git_worktree_summary
from tools.subagents import set_subagent_runtime
from tools.todo import get_thread_todos, set_current_thread, set_thread_todos
from compaction import (
    compact_messages_progressively,
    format_compaction_overlay_note,
    resolve_token_count,
    resolve_usable_context_budget,
)
# 1. Define the AgentState
class AgentState(TypedDict, total=False):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    approval_declined: bool
    todos: list[dict]
    agent_mode: str
    last_reflected_message_index: int
    compacted_messages: list[BaseMessage]
    compaction_message_count: int
    force_compact: bool
    last_input_tokens: int

# 2. Build the Graph
# Approval handler: given (tool_name, args) returns a normalized decision in
# {"yes", "session", "always", "no", "never"}. When None, gated tools are denied.
ApprovalHandler = Callable[[str, dict], Awaitable[str]]


def build_graph(
    model: ChatOpenRouter,
    tools=None,
    thread_id: str = "thread",
    agent_mode: str | None = None,
    git_available: bool | None = None,
    checkpointer: Any | None = None,
    approval_handler: ApprovalHandler | None = None,
    question_handler: QuestionHandler | None = None,
) -> Runnable:
    # define the mode
    resolved_mode = (agent_mode or "normal").lower()
    repo_has_git = is_git_repo() if git_available is None else git_available

    # Runtime tool holder. The bound tool set can grow mid-session when MCP tools
    # are loaded (search_tools/add_tools or /mcp). We hot-rebind the model when the
    # tools generation changes instead of recompiling the graph, so MemorySaver
    # state is preserved. Loading a tool costs exactly one prefix-cache break (the
    # _stable_prefix key includes tool_names), then the new set re-stabilizes.
    # Subagents pass an explicit `tools` list and keep a fixed set (is_dynamic=False).
    is_dynamic = tools is None
    runtime: dict[str, Any] = {}

    def _sync_runtime(force: bool = False) -> None:
        if runtime and not force:
            if not is_dynamic:
                return
            if runtime.get("generation") == tools_generation():
                return
        candidate_tools = ALL_TOOLS if tools is None else tools
        active = select_tools_for_session(repo_has_git, candidate_tools)
        runtime["active_tools"] = active
        runtime["tool_map"] = {t.name: t for t in active}
        runtime["tool_names"] = list(runtime["tool_map"])
        runtime["bound_model"] = model.bind_tools(active)
        runtime["generation"] = tools_generation()

    _sync_runtime(force=True)
    all_skills = load_skills()

    # create buffers
    # sticky skills are skills that are always loaded mainly the SKILL.md file (exluding the references, scripts etc.)
    sticky_skill_names: set[str] = set()
    prefix_cache: dict[str, Any] = {}

    # tasks to reflect on the conversation and update sessions/mem_<thread_id>.md
    reflection_tasks: set[asyncio.Task] = set()
    compaction_model = create_compaction_model(thread_id)

    # 3. Define the Agent Node
    async def agent_node(state: AgentState) -> AgentState:
        messages = list(state.get("messages", []))

        # hot-rebind the model if MCP tools were loaded since the last turn
        _sync_runtime()

        # checking for /compact; if yes then clear the prefix cache
        if state.get("force_compact"):
            prefix_cache.clear()

        user_input = next((m.content for m in reversed(messages) if m.type == "human"), "")
        
        previous_skill_key = tuple(sorted(sticky_skill_names))
        active_skills = select_sticky_skills(user_input, all_skills, sticky_skill_names)
        
        # check the cache key to see if the sticky skills have changed
        if tuple(sorted(sticky_skill_names)) != previous_skill_key:
            prefix_cache.clear()

        # build the system message
        system = SystemMessage(content=_stable_prefix(active_skills))

        # get the effective conversation
        conversation = _effective_conversation(messages, state)

        # set the thread id and todos in the _todo_store
        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))
        
        # estimate the total input tokens (- L3 working state) for compaction requirements
        estimated_model_input_tokens = resolve_token_count(
            [system] + conversation,
            known_input_tokens=state.get("last_input_tokens") or None,
        )

        # compact the messages if needed
        compaction = await compact_messages_progressively(
            conversation,
            known_input_tokens=estimated_model_input_tokens,
            summary_model=compaction_model,
            force=bool(state.get("force_compact")),
            model_name=settings.model_name,
            thread_id=thread_id,
        )
        if compaction.compacted:
            conversation = compaction.messages

        # format the compaction note
        compaction_note = format_compaction_overlay_note(
            compaction,
            had_stored_compaction=bool(state.get("compacted_messages")),
        )

        current_mode = (state.get("agent_mode") or resolved_mode).lower()
        git_snapshot = git_worktree_summary() if repo_has_git else ""
        session_memory = load_session_memory(thread_id)

        # build the working state overlay (L3): Git snapshot, compaction note, todos, session memory
        overlay = build_working_state_overlay(
            current_mode,
            todos=render_todos(state.get("todos", [])),
            session_memory=session_memory,
            git_snapshot=git_snapshot,
            compaction_note=compaction_note,
        )

        # append the L3 working state as a dedicated ephemeral message at the tail
        model_messages = [system] + _with_working_state_tail(conversation, overlay)
        response: AIMessage = await runtime["bound_model"].ainvoke(model_messages)

        updates: AgentState = {
            "messages": [response],
            "approval_declined": False,
            "force_compact": False,
        }

        # track conversation for auto-compaction
        if compaction.compacted:
            updates["compacted_messages"] = conversation
            updates["compaction_message_count"] = len(messages)
            updates["last_input_tokens"] = 0 # after compaction, we don't know the exact input tokens. It falls back to estimation.

        # track usage after every API call
        # langchain clients track usage metadata for billing and analytics
        # tracks -> input tokens, output tokens, total tokens, input_token_details (contains cache_read), output_token_details (contains reasoning tokens)
        if response.usage_metadata:
            updates["last_input_tokens"] = _usage_value(
                response.usage_metadata, "input_tokens", "prompt_tokens"
            )
            usage = cost_tracker.add(
                response.usage_metadata,
                settings.model_name,
                response.response_metadata or {},
            )
            usage_event = {"kind": "usage", "model": settings.model_name}
            if usage:
                usage_event.update(usage)
            append_event(thread_id, usage_event)

        # after getting the response of human message, schedule reflection if token delta exceeds threshold
        user_count = sum(1 for message in messages if message.type == "human")
        schedule_reflection(state, messages + [response], user_count, updates)

        # picks up the index from the previous reflection run
        completed_index = consume_reflection_message_index(thread_id)
        if completed_index is not None:
            updates["last_reflected_message_index"] = completed_index

        content = response.content
        # append the assistant message to the threads db - events table
        append_event(
            thread_id,
            {
                "kind": "assistant",
                "content": content if isinstance(content, (str, type(None))) else str(content),
                "tool_calls": response.tool_calls or [],
            },
        )
        return updates

    # 4. Define the Approval Gate Node
    async def approval_gate(state: AgentState) -> AgentState:
        last = state["messages"][-1]
        calls = extract_tool_calls(last)
        gated = [(name, args, call_id) for name, args, call_id in calls if _needs_approval(name, args)]
        if not gated:
            return {"approval_declined": False}

        for name, args, _ in gated:
            if approval_handler is None:
                append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
                return _denied_messages(calls, f"Approval required but no handler configured: {name}")

            decision = await approval_handler(name, args)

            if decision == "always":
                rule = default_rule_for(name, args)
                persist_rule(rule, "allow")
                append_event(thread_id, {"kind": "approval", "tool": name, "decision": "always", "rule": rule})
                continue
            if decision == "session":
                rule = default_rule_for(name, args)
                persist_rule(rule, "allow", scope="session")
                append_event(thread_id, {"kind": "approval", "tool": name, "decision": "session", "rule": rule})
                continue
            if decision == "never":
                rule = default_rule_for(name, args)
                persist_rule(rule, "deny")
                append_event(thread_id, {"kind": "approval", "tool": name, "decision": "never", "rule": rule})
                return _denied_messages(calls, f"Denied by persisted permission rule: {rule}")
            if decision == "yes":
                append_event(thread_id, {"kind": "approval", "tool": name, "decision": "yes"})
                continue
            # any other value is treated as a one-off decline
            append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
            return _denied_messages(calls, f"Denied by user approval: {name}")

        return {"approval_declined": False}

    # 5. Define the Tools Node
    async def tools_node(state: AgentState) -> AgentState:
        # get the last AIMessage and extract the tool calls
        last = state["messages"][-1]
        calls = extract_tool_calls(last) # returns list of tuples (name, args, call_id)
        if not calls:
            return {"messages": []}

        # set the thread id and todos in the _todo_store and set the subagent runtime
        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))
        set_subagent_runtime(model, thread_id)
        set_question_runtime(question_handler)

        # store tool results in a list of ToolMessage objects
        results: list[ToolMessage] = []
        current_mode = (state.get("agent_mode") or resolved_mode).lower()
        for name, args, call_id in calls:
            if current_mode == "plan" and not is_read_only_tool_call(name, args):
                content = "Unavailable in plan mode. Switch to /act to execute."
                # hidden=True keeps the rejection in state (the model sees it and adapts)
                # but the CLI render layer skips it, so the user is not shown the noise.
                results.append(
                    ToolMessage(
                        tool_call_id=call_id,
                        name=name,
                        content=content,
                        additional_kwargs={"hidden": True},
                    )
                )
                append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="mode_gated"))
                continue

            # permission check: "ask", "allow", "deny"
            decision, rule = check_with_rule(name, args)
            if decision == "deny":
                content = f"Denied by permission rule: {rule}"
                results.append(ToolMessage(tool_call_id=call_id, name=name, content=content))
                append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="denied"))
                continue

            # run the preToolUse hook
            # run_hooks() returns (True/False, message) and can be used before and after tool use
            ok, hook_msg = run_hooks("preToolUse", {"tool": name, "args": args, "thread_id": thread_id})
            # some hooks may veto the tool use
            if not ok:
                content = f"Hook veto: {hook_msg}"
                results.append(ToolMessage(tool_call_id=call_id, name=name, content=content))
                append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="denied"))
                continue

            # run the tool and measure the duration
            started = time.time()
            tool = runtime["tool_map"].get(name)
            try:
                if tool is None:
                    result = f"Error: unknown tool {name}"
                elif getattr(tool, "is_async", False) or getattr(tool, "coroutine", None) is not None:
                    result = await tool.ainvoke(args)
                else:
                    result = tool.invoke(args)
            except Exception as exc:
                result = f"Error: {exc}"
            duration = int((time.time() - started) * 1000)

            content = str(result)
            
            # run the postToolUse hook
            run_hooks(
                "postToolUse",
                {"tool": name, "args": args, "result": content, "thread_id": thread_id},
            )

            # append the ToolMessage to the results list and append the event to the event log
            results.append(ToolMessage(tool_call_id=call_id, name=name, content=content))
            append_event(thread_id, _tool_event(name, args, content, duration, call_id=call_id))

        todos_after = get_thread_todos(thread_id)
        return {"messages": results, "todos": todos_after}

    # 6. Define the Conditional Routers
    async def route_after_agent(state: AgentState) -> Literal["approval_gate", "tools", END]:
        last = state["messages"][-1]
        calls = extract_tool_calls(last)
        if not calls:
            return END
        current_mode = (state.get("agent_mode") or resolved_mode).lower()
        if current_mode == "plan" and any(not is_read_only_tool_call(name, args) for name, args, _ in calls):
            return "tools"
        if any(_needs_approval(name, args) for name, args, _ in calls):
            return "approval_gate"
        return "tools"

    async def route_after_approval(state: AgentState) -> Literal["agent", "tools"]:
        if state.get("approval_declined"):
            return "agent"
        return "tools"


    def _stable_prefix(active_skills: list[dict[str, Any]]) -> str:
        """
        Building a key to know when to re-build the prefix cache.
        - this stable prefix only considers L1 and L2 context blocks (part of system message). 
        - L3 is changing almost on every human message so it is excluded from the key.
        - this key reduces context re-builds on every turn.
        
        key should contain: 
        - persona/profile, 
        - tool names (tuple), sticky skills (tuple), git availability (bool), 
        - ness memory, user memory <format: (bool, mtime_ns, size)>

        agent mode is deliberately excluded: the full tool set is always bound and
        the mode block lives in the ephemeral L3 overlay, so mode switches must not
        invalidate the cached prefix.
        """
        active_tools = runtime["active_tools"]
        key = (
            DEFAULT_PERSONA_ID,
            tuple(runtime["tool_names"]),
            tuple(sorted(str(skill.get("name", "")) for skill in active_skills)),
            repo_has_git,
            ness_key(),
            user_memory_key(),
        )
        # manual cache verification.
        if prefix_cache.get("key") != key:
            prefix_cache["content"] = "\n\n".join(
                [
                    build_l0(active_tools),
                    build_l1(
                        DEFAULT_PERSONA,
                        active_tools,
                        load_user_memory(),
                        load_ness_memory(),
                    ),
                    build_project_context_block(
                        load_repo_context(),
                        active_skills,
                        repo_has_git,
                    ),
                ]
            ).strip()
            prefix_cache["key"] = key
        return str(prefix_cache["content"])
    
    async def reflect_in_background(
        messages: list[BaseMessage],
        user_count: int,
        *,
        since_index: int,
        todos: list[dict],
    ) -> None:
        reflection_client = create_reflection_model(thread_id)
        await run_reflection_gate(
            thread_id,
            messages,
            reflection_client,
            user_count,
            last_reflected_message_index=since_index,
            todos=render_todos(todos),
        )

    def schedule_reflection(
        state: AgentState,
        messages: list[BaseMessage],
        user_count: int,
        updates: AgentState,
    ) -> None:

        # decides whether to schedule a new reflection task
        # check with the async lock if reflection is already running
        if is_reflection_running(thread_id):
            return

        # estimate the token delta since the last reflection run
        since_index = int(state.get("last_reflected_message_index", 0) or 0)
        delta = _reflection_token_delta(messages, since_index)
        threshold = _reflection_token_threshold()
        if delta < threshold:
            return

        todos = list(state.get("todos", []))

        # create reflection task in background
        task = asyncio.create_task(
            reflect_in_background(
                messages,
                user_count,
                since_index=since_index,
                todos=todos,
            ),
            name=f"reflection-{thread_id}-{user_count}",
        )
        # add the task to the set of reflection tasks
        reflection_tasks.add(task)
        # remove the task from the set of reflection tasks when it is done
        task.add_done_callback(reflection_tasks.discard)

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("approval_gate", approval_gate)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", route_after_agent)
    graph.add_conditional_edges("approval_gate", route_after_approval)
    graph.add_edge("tools", "agent")
    return graph.compile(checkpointer=checkpointer or MemorySaver())


def _needs_approval(name: str, args: dict) -> bool:
    if not settings.enable_approval:
        return False
    decision, _ = check_with_rule(name, args)
    if decision in {"allow", "deny"}:
        return False
    return is_destructive_tool_call(name, args)


def _denied_messages(calls: list[tuple[str, dict, str]], content: str) -> AgentState:
    return {
        "messages": [
            ToolMessage(tool_call_id=call_id, name=name, content=content)
            for name, _, call_id in calls
        ],
        "approval_declined": True,
    }


def _effective_conversation(messages: list[BaseMessage], state: AgentState) -> list[BaseMessage]:
    # builds the effective message list at every turn
    # if compaction exists then we need compacted + raw[source_count:] else just raw system message
    compacted = list(state.get("compacted_messages", []))
    source_count = int(state.get("compaction_message_count", 0) or 0)
    raw = [m for m in messages if m.type != "system"]
    if compacted and 0 <= source_count <= len(raw): 
        return compacted + raw[source_count:]
    return raw


def _with_working_state_tail(messages: list[BaseMessage], overlay: str) -> list[BaseMessage]:
    # Append the L3 working state as a dedicated ephemeral message at the TAIL.
    # - never written back to state: it lives only in this transient model_messages list
    if not overlay.strip():
        return list(messages)
    block = f"<working-state>\n{overlay.strip()}\n</working-state>"
    return list(messages) + [HumanMessage(content=block)]


def _reflection_token_delta(messages: list[BaseMessage], since_index: int) -> int:
    """Estimate tokens in messages not yet covered by the last reflection run."""
    since_index = max(0, since_index)
    recent = list(messages)[since_index:]
    if not recent:
        return 0
    return resolve_token_count(recent, known_input_tokens=None)


def _reflection_token_threshold() -> int:
    """
    Token delta required before scheduling background reflection.
    budget = usable context window - reserve tokens
    delta = budget * ratio
    """
    ratio = float(settings.reflection_token_ratio or 0)
    if ratio <= 0:
        return 0
    # get the usable context window from compaction
    budget = resolve_usable_context_budget(model_name=settings.model_name)
    return int(budget * ratio)


def _tool_event(
    name: str,
    args: dict,
    result: str,
    duration: int,
    *,
    call_id: str = "",
    exit_status: str | None = None,
) -> dict:
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


def _result_status(result: str) -> str | None:
    for line in result.splitlines()[:8]:
        if line.startswith("status="):
            status = line.removeprefix("status=").strip()
            return status or None
    return None
