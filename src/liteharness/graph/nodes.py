from __future__ import annotations

import asyncio, json, time, warnings
from pathlib import Path
from typing import Any, Literal, Mapping
from langchain_core.messages import AIMessage, SystemMessage, ToolMessage, BaseMessage
from liteharness.graph.state import AgentState
from liteharness.graph.helpers import (
    _effective_conversation,
    _with_working_state_tail,
    _needs_approval,
    _denied_messages,
    _reflection_token_delta,
    extract_tool_calls,
    _tool_event,
)
from liteharness.compaction import (
    compact_messages_progressively,
    format_compaction_overlay_note,
    resolve_token_count,
    resolve_usable_context_budget,
)
from liteharness.reflection import (
    is_reflection_running,
    consume_reflection_message_index,
    run_reflection_gate,
)
from liteharness.tools.todo import set_current_thread, set_thread_todos
from liteharness.workspace.git_context import git_worktree_summary
from liteharness.tracing.semconv import (
    CACHE_HIT_RATE,
    CACHE_READ_TOKENS,
    COST_USD,
    GEN_AI_COMPLETION,
    GEN_AI_OPERATION_NAME,
    GEN_AI_PROMPT,
    GEN_AI_SYSTEM,
    GEN_AI_SYSTEM_VALUE,
    GEN_AI_TOOL_CALL_ARGUMENTS,
    GEN_AI_TOOL_CALL_RESULT,
    KIND_CLIENT,
    INPUT_TOKENS,
    LLM_CALL,
    MODEL_NAME,
    OUTPUT_TOKENS,
    TOOL_ARGS,
    TOOL_DURATION_MS,
    TOOL_ERROR,
    TOOL_EXEC,
    TOOL_EXIT_STATUS,
    TOOL_NAME,
)
from liteharness.tracing.messages import (
    serialize_completion,
    serialize_messages,
    truncate_for_span,
)
from liteharness.tracing import TokenUsage
from liteharness.types import UsageEvent
from liteharness.tools.todo import render_todos

class NodesRuntime:
    """Mutable container that carry the states between the nodes.
    These states need to outlive single node call but cannot remain in the AgentState."""
    def __init__(self, config, *, thread_id, agent_mode = "act", git_available, metadata = None):
        self.cfg = config
        self.thread_id = thread_id
        self.resolved_mode = (agent_mode or "act").lower()
        self.repo_has_git = git_available == True
        self._last_sections: dict[str, str] = {}
        self.reflection_tasks: set[asyncio.Task] = set()
        self.metadata: Mapping[str, Any] = metadata if metadata is not None else {}

def make_nodes(config, *, thread_id, agent_mode = "act", git_available = None, metadata = None) -> NodesRuntime:
    rt = NodesRuntime(config, thread_id=thread_id, agent_mode=agent_mode, git_available=git_available, metadata=metadata)
    # objects of the backends created in the NessAgentConfig
    tools_reg = config.tool_registry
    skills_loader = config.skill_loader
    perms = config.permission_store
    hooks = config.hook_runner
    cost = config.cost_tracker
    memory = config.memory_store
    persist = config.thread_store
    prompts = config.prompts
    overlay_provider = config.overlay
    task_prompts = config.task_prompts
    options = config.options
    tracer = config.tracer
    all_skills = skills_loader.load()

    main_model = config.model
    compaction_model = config.compaction_model or config.model

    async def agent_node(state: AgentState) -> AgentState:
        """The main agent node that handles the agent's logic."""

        # hot-rebind tools if new tools were loaded since the last turn
        tools_reg.sync()
        messages = list(state.get("messages", []))

        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))

        # load the components for the L1 prompt layer
        user_mem = memory.load_user() if not memory.disabled else ""
        proj_mem = memory.load_project() if not memory.disabled else ""
        skill_catalog = skills_loader.render_catalog(all_skills)

        # build the full system message with stable prefix
        # the stable prefix is cached in the prompts object. It only builds if the key is different.
        system = SystemMessage(content=prompts.build_stable_prefix(
            tools_reg.active_tools,
            user_memory=user_mem,
            project_memory=proj_mem,
            skill_catalog=skill_catalog,
            git_available=rt.repo_has_git,
            metadata=rt.metadata,
            tool_catalog_groups=[(l, frozenset(g)) for l, g in tools_reg.tool_catalog_groups()],
            mcp_catalog=tools_reg.mcp_catalog(),
            deferred_mcp=tools_reg.deferred_mcp_summary()
        ))

        conversation = _effective_conversation(messages, state)
        model_name = getattr(config.model, "model", "") or getattr(config.model, "model_name", "")

        # estimate the total input tokens (- L3 working state) for compaction requirements.
        # run in a thread: tokenizer pass over the full conversation is sync CPU and would
        # stall the event loop (and the TUI working spinner) before the first token lands.
        token_estimate = await asyncio.to_thread(resolve_token_count, [system] + conversation, known_input_tokens=state.get("last_input_tokens") or None)

        compaction = await compact_messages_progressively(
            conversation,
            known_input_tokens=token_estimate,
            summary_model=compaction_model,
            force=bool(state.get("force_compact")),
            model_name=model_name,
            thread_id=thread_id,
            options=options,
            cost_tracker=cost,
            persistence=persist,
            tracer=tracer,
            tracing=config.tracing,
            compaction_prompt=task_prompts.compaction
        )
        if compaction.compacted: 
            conversation = compaction.messages

        compaction_note = format_compaction_overlay_note(
            compaction,
            options=options,
            had_stored_compaction=bool(state.get("compacted_messages")),
            model_name=model_name,
        )

        cwd = options.project_root or Path.cwd()
        git_snapshot = (
            await asyncio.to_thread(git_worktree_summary, cwd)
            if rt.repo_has_git else ""
        )

        # L3 Overlay
        if overlay_provider is not None:
            from liteharness.context.overlay import OverlayContext, render_overlay_delta
            overlay_context = OverlayContext(
                thread_id=thread_id,
                agent_mode=(state.get("agent_mode") or rt.resolved_mode),
                messages=conversation,
                todos=state.get("todos", []),
                session_memory=memory.load_session(thread_id) if not memory.disabled else "",
                compaction_note=compaction_note,
                mode_switch=state.get("mode_switch") or "",
                metadata=rt.metadata,
                git_snapshot=git_snapshot,
                git_available=rt.repo_has_git,
                activate_skills=list(state.get("activate_skills", [])),
                loaded_skills=list(state.get("loaded_skills", [])),
            )
            sections = overlay_provider.sections(state, overlay_context) or {}
        else:
            sections = {}

        # Fresh turn (last conversation message is a HumanMessage) or post-compaction
        # (model context was rewritten) -> inject the FULL overlay so plan-mode
        # instructions are (re)established.  
        # For Tool loop -> inject only the per-section delta,
        # skipping the static plan_mode block (already on the user message).
        # Plain join only — _with_working_state_tail wraps <system-reminder>.
        is_fresh = bool(conversation) and conversation[-1].type == "human"
        if is_fresh or compaction.compacted:
            overlay = "\n\n".join(sections.values())
        else:
            overlay = render_overlay_delta(sections, rt._last_sections, skip=frozenset({"plan_mode"}))
        
        rt._last_sections.clear()
        rt._last_sections.update(sections)

        bound_model = tools_reg.bind_model(main_model)
        # reset some states
        updates: AgentState = {
            "messages": [],
            "approval_declined": False,
            "force_compact": False,
            "activate_skills": [],
            "mode_switch": "",
        }
        last_input_tokens = token_estimate
        llm_attrs = {
            MODEL_NAME: model_name,
            GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
            GEN_AI_OPERATION_NAME: "chat",
        }
        # Build the exact invoke payload once (system + working-state tail injected ephemerally).
        invoke_messages = [system] + _with_working_state_tail(conversation, overlay)
        
        with tracer.start_span(LLM_CALL, attributes=llm_attrs, kind=KIND_CLIENT) as llm_span:

            if config.tracing.capture_messages:
                llm_span.set_attribute(GEN_AI_PROMPT, serialize_messages(invoke_messages))
            
            response: AIMessage = await bound_model.ainvoke(invoke_messages)
            
            # update the AgentState
            updates["messages"] = [response]

            if config.tracing.capture_messages:
                llm_span.set_attribute(GEN_AI_COMPLETION, serialize_completion(response))

            # track usage after every API call
            # langchain clients track usage metadata for billing and analytics
            # tracks -> input tokens, output tokens, total_tokens, input_token_details
            # (contains cache_read), output_token_details (contains reasoning tokens)
            if response.usage_metadata:
                usage: TokenUsage | None = cost.add(
                    response.usage_metadata,
                    model_name,
                    response.response_metadata or {},
                )
                if usage is not None:
                    llm_span.set_attribute(INPUT_TOKENS, usage.input_tokens)
                    llm_span.set_attribute(OUTPUT_TOKENS, usage.output_tokens)
                    llm_span.set_attribute(CACHE_READ_TOKENS, usage.cached_input_tokens)
                    llm_span.set_attribute(CACHE_HIT_RATE, usage.cache_hit_rate)
                    if usage.cost_usd is not None:
                        llm_span.set_attribute(COST_USD, usage.cost_usd)
                    last_input_tokens = usage.input_tokens

                usage_event: dict[str, Any] = {"kind": "usage", "model": model_name}
                if usage is not None:
                    usage_event.update(usage.as_dict())
                persist.append_event(thread_id, usage_event)

                if usage is not None and config.on_usage:
                    config.on_usage(UsageEvent(
                        model=model_name,
                        input_tokens=usage.input_tokens,
                        uncached_input_tokens=usage.uncached_input_tokens,
                        cached_input_tokens=usage.cached_input_tokens,
                        output_tokens=usage.output_tokens,
                        cost_usd=usage.cost_usd,
                    ))

        # track conversation for auto-compaction
        if compaction.compacted:
            updates["compacted_messages"] = conversation
            updates["compaction_message_count"] = len(messages)
            updates["last_input_tokens"] = 0  # next turn re-estimates if no usage_metadata
        else:
            updates["last_input_tokens"] = last_input_tokens

        persist.append_event(
            thread_id, {
                "kind": "assistant", 
                "content": str(response.content),
                "tool_calls": response.tool_calls or []
            }
        )
        _maybe_schedule_reflection(rt, state, messages + [response], model_name)
        ci = consume_reflection_message_index(thread_id)
        if ci is not None: 
            updates["last_reflected_message_index"] = ci
        
        return updates


    async def approval_gate(state: AgentState) -> AgentState:
        """The approval gate node that handles the approval logic."""
        calls = extract_tool_calls(state["messages"][-1])
        gated = [(n, a, cid) for n, a, cid in calls if _needs_approval(n, a, options, perms, tools_reg)]
        
        if not gated: 
            return {"approval_declined": False}
        
        ah = config.approval_handler
        for name, args, _ in gated:
            if ah is None:
                persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
                return _denied_messages(calls, f"Approval required but no handler configured: {name}")
            
            decision = await ah(name, args)

            if decision == "always":
                rule = perms.default_rule_for(name, args)
                perms.persist_rule(rule, "allow")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "always", "rule": rule}
                )
                continue
            if decision == "session":
                rule = perms.default_rule_for(name, args)
                perms.persist_rule(rule, "allow", scope="session")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "session", "rule": rule}
                )
                continue
            if decision == "never":
                rule = perms.default_rule_for(name, args)
                perms.persist_rule(rule, "deny")
                persist.append_event(
                    thread_id, {"kind": "approval", "tool": name, "decision": "never", "rule": rule}
                )
                return _denied_messages(calls, f"Denied by persisted permission rule: {rule}")
            if decision == "yes":
                persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "yes"})
                continue
            warnings.warn(
                f"Unknown approval decision {decision!r} from {ah!r} for tool {name!r} — "
                f"treating as denied",
                stacklevel=2,
            )
            persist.append_event(thread_id, {"kind": "approval", "tool": name, "decision": "no"})
            return _denied_messages(calls, f"Denied by user approval: {name}")
        
        return {"approval_declined": False}

    async def tools_node(state: AgentState) -> AgentState:
        """The tools node that handles the tool calls and tool results logic."""
        from liteharness.tools.ask import set_question_runtime
        from liteharness.tools.subagents import set_subagent_runtime
        from liteharness.tools.todo import set_current_thread, set_thread_todos, get_thread_todos

        # get the last AIMessage and extract the tool calls
        calls = extract_tool_calls(state["messages"][-1])
        if not calls:
            return {"messages": []}

        set_subagent_runtime(main_model, thread_id)
        set_question_runtime(config.question_handler)
        set_current_thread(thread_id)
        set_thread_todos(thread_id, list(state.get("todos", [])))

        # store tool results in a list of ToolMessage objects
        results: list[ToolMessage] = []
        cur_mode = (state.get("agent_mode") or rt.resolved_mode).lower()
        raw_user_seq = state.get("current_user_seq")
        user_seq = int(raw_user_seq) if raw_user_seq is not None else None
        newly_loaded_names: set[str] = set()

        for name, args, call_id in calls:
            # if the mode is plan and the tool is not read-only then return the denied messages
            if config.modes and cur_mode == "plan" and not tools_reg.is_read_only(name, args):
                content = "Unavailable in plan mode. Switch to act mode to run state-changing tools."
                results.append(ToolMessage(
                    tool_call_id=call_id,
                    name=name,
                    content=content,
                    additional_kwargs={"hidden": True}
                ))
                persist.append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="mode_gated"))
                continue

            # if the tool is denied by a permission rule then return the denied messages
            decision, rule = perms.check_with_rule(name, args)
            if decision == "deny":
                content = f"Denied by permission rule: {rule}"
                results.append(ToolMessage(tool_call_id=call_id, name=name, content=content))
                persist.append_event(thread_id, _tool_event(name, args, content, 0, call_id=call_id, exit_status="denied"))
                continue

            # run the preToolUse hook
            ok, msg = hooks.run("preToolUse", {"tool": name, "args": args, "thread_id": thread_id})
            # if the hook vetoed the tool use then return the denied messages
            if not ok:
                results.append(ToolMessage(tool_call_id=call_id, name=name, content=f"Hook veto: {msg}"))
                persist.append_event(thread_id, _tool_event(name, args, msg, 0, call_id=call_id, exit_status="denied"))
                continue

            # invoke the tool
            tmap = tools_reg.tool_map().get(name)
            tool_attrs: dict[str, Any] = {
                TOOL_NAME: name,
                GEN_AI_SYSTEM: GEN_AI_SYSTEM_VALUE,
                GEN_AI_OPERATION_NAME: "execute_tool",
            }
            capture_msgs = config.tracing.capture_messages
            if config.tracing.capture_tool_args:
                tool_attrs[TOOL_ARGS] = str(args)[:500]

            t0 = time.monotonic()
            with tracer.start_span(
                TOOL_EXEC.format(name=name), attributes=tool_attrs, kind=KIND_CLIENT
            ) as tool_span:
                if capture_msgs:
                    # Canonical JSON form parsed by Langfuse/Arize as tool input.
                    tool_span.set_attribute(GEN_AI_TOOL_CALL_ARGUMENTS, json.dumps(args, default=str))
                try:
                    if tmap is None:
                        result = f"Error: unknown tool {name}"
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "unknown_tool")
                    elif getattr(tmap, "is_async", False) or getattr(tmap, "coroutine", None) is not None:
                        result = await tmap.ainvoke(args)
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "ok")
                    else:
                        result = await asyncio.to_thread(tmap.invoke, args)
                        tool_span.set_attribute(TOOL_EXIT_STATUS, "ok")
                    tool_span.set_attribute(TOOL_ERROR, False)
                except Exception as exc:
                    tool_span.record_exception(exc)
                    tool_span.set_attribute(TOOL_ERROR, True)
                    tool_span.set_attribute(TOOL_EXIT_STATUS, "exception")
                    result = f"Error: {exc}"
                tool_span.set_attribute(TOOL_DURATION_MS, int((time.monotonic() - t0) * 1000))
                if capture_msgs:
                    # Tool results can be MBs — truncate
                    # to keep OTLP batches under the SDK's ~5MB limit
                    tool_span.set_attribute(
                        GEN_AI_TOOL_CALL_RESULT,
                        truncate_for_span(str(result), config.tracing.max_message_length),
                    )
            dur = int((time.monotonic() - t0) * 1000)
            content = str(result)

            # run the postToolUse hook
            _ok, hook_msg = hooks.run("postToolUse", {"tool": name, "args": args, "result": content, "thread_id": thread_id})
            if hook_msg:
                content = hook_msg + "\n\n" + content if content.strip() else hook_msg
            results.append(ToolMessage(tool_call_id=call_id, name=name, content=content))
            # append the ToolMessage to the results list and append the event to the event log
            persist.append_event(thread_id, _tool_event(name, args, content, dur, call_id=call_id))

            # Record mutated paths for rollback (surgical file restore).
            # Only run on a non-error result so we don't pollute the checkpoint
            # with paths the tool never touched. ``current_user_seq is None``
            # means there is no live user turn in state (subagent / pure-SDK
            # path) -> skip.
            if user_seq is not None and config.on_file_mutation and not str(content).startswith("Error:"):
                config.on_file_mutation(thread_id, user_seq, name, args)

            # Track skills loaded via skill_view for the L3 overlay
            if name == "skill_view" and not str(content).startswith("Error:"):
                sk_name = str(args.get("name", ""))
                if sk_name and sk_name in all_skills:
                    newly_loaded_names.add(sk_name)
        # Merge newly-loaded skills into persistent loaded_skills state
        existing = list(state.get("loaded_skills", []))
        existing_names = {s.get("name", "") for s in existing}
        for sk_name in sorted(newly_loaded_names):
            if sk_name not in existing_names and sk_name in all_skills:
                sk = all_skills[sk_name]
                existing.append({
                    "name": sk.get("name", sk_name),
                    "description": sk.get("description", ""),
                    "path": sk.get("source", ""),
                })
        return {"messages": results, "todos": get_thread_todos(thread_id), "loaded_skills": existing}

    async def route_after_agent(state) -> Literal["approval_gate", "tools", "__end__"]:
        from langgraph.graph import END

        calls = extract_tool_calls(state["messages"][-1])
        
        if not calls: 
            return END

        cur = (state.get("agent_mode") or rt.resolved_mode).lower()

        # if the mode is plan and the tool is not read-only then return the tools node
        if config.modes and cur == "plan" and any(not tools_reg.is_read_only(n, a) for n, a, _ in calls): 
            return "tools"

        # if the approval is enabled and the tool needs approval then return the approval gate node
        if options.enable_approval and any(_needs_approval(n, a, options, perms, tools_reg) for n, a, _ in calls):
            return "approval_gate"
        
        return "tools"

    async def route_after_approval(state) -> Literal["agent", "tools"]:
        if state.get("approval_declined"):
            return "agent"
        return "tools"

    rt.agent_node = agent_node
    rt.approval_gate = approval_gate
    rt.tools_node = tools_node
    rt.route_after_agent = route_after_agent
    rt.route_after_approval = route_after_approval
    return rt



def _maybe_schedule_reflection(
    rt: NodesRuntime, 
    state: AgentState, 
    messages: list[BaseMessage], 
    model_name: str
) -> None:
    """Schedules a reflection task if the reflection token delta is greater than the reflection token ratio."""
    
    if is_reflection_running(rt.thread_id): 
        return
    since = int(state.get("last_reflected_message_index", 0) or 0) # get the last reflected message index
    delta = _reflection_token_delta(messages, since)
    ratio = float(rt.cfg.options.reflection_token_ratio or 0)
    if ratio <= 0: 
        return
    
    budget = resolve_usable_context_budget(model_name, rt.cfg.options)
    if delta < int(budget * ratio): 
        return
    todos = list(state.get("todos", [])) # get the todos
    
    async def _bg():
        await run_reflection_gate(
            rt.thread_id,
            messages,
            rt.cfg.reflection_model or rt.cfg.model,
            sum(1 for m in messages if m.type == "human"),
            last_reflected_message_index=since,
            todos=render_todos(todos),
            memory=rt.cfg.memory_store,
            persistence=rt.cfg.thread_store,
            task_prompts=rt.cfg.task_prompts,
            tracer=rt.cfg.tracer,
            tracing=rt.cfg.tracing,
        )
    
    task = asyncio.create_task(_bg(), name=f"reflection-{rt.thread_id}")
    rt.reflection_tasks.add(task)
    task.add_done_callback(rt.reflection_tasks.discard)