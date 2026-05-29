import asyncio, difflib, os, time
from tkinter import E
from typing import Literal, TypedDict, Annotated, Sequence
import operator
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import (
    HumanMessage, AIMessage, SystemMessage, ToolMessage, BaseMessage,
)
from pathlib import Path
from langchain_core.runnables import Runnable
from langchain_openrouter import ChatOpenRouter
from config import settings, cost_tracker
from memory import load_project_context
from permissions import check, persist_rule
from hooks import run_hooks
from session import append_event
from parsers import extract_tool_calls
from prompt import get_system_prompt
from skill_loader import load_skills, inject_skills, select_skills
from utils import trim_messages_smart, _needs_approval, _preview_diff
from tools import ALL_TOOLS, TOOL_MAP, TOOL_NAMES, DESTRUCTIVE_TOOLS

# 1. Define the AgentState

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    approval_declined: bool
    todos: list[dict]
    subagent_dept: int


# 2. Define the Graph Nodes

def build_graph(model:ChatOpenRouter, tools=None) -> Runnable:
    tools = tools or ALL_TOOLS
    tool_map = {t.name: t for t in tools}
    tool_names = list(tool_map.keys())
    bound_model = model.bind_tools(tools) if settings.mode == "json" else model

    # 3. Define the AgentNode
    async def agent_node(state:AgentState) -> AgentState:
        messages = list(state["messages"])
        user_input = next((m.content for m in reversed(messages) if m.type == "human"), "")
        
        # load the skills
        skills = load_skills()
        active = select_skills(str(user_input), skills)
        base = inject_skills(get_system_prompt(settings.mode), active)
        
        # build the system message
        has_assistant = any(m.type in ("ai", "tool", "assistant") for m in messages)
        if not has_assistant:
            ctx = load_project_context()
            sys_msg = SystemMessage(content=f"{base}\n\nProject context:\n{ctx}")
        else:
            sys_msg = SystemMessage(content=base)

        messages = [sys_msg] + [m for m in messages if m.type != "system"]
        # trim the messages
        messages = await trim_messages_smart(messages, settings.max_tokens)

        response = await bound_model.ainvoke(messages)
        # track the usage after invoke
        if response.usage_metadata:
            cost_tracker.add(response.usage_metadata, settings.model_name)
            append_event("thread", {"kind": "usage", "model": settings.model_name})

        return {"messages": [response], "approval_decline": False}

    # 4. Define the Approval Gate Node
    async def approval_gate(state:AgentState) -> AgentState:
        # get the last message
        last = state["messages"][-1]
        # extract the last tool calls
        calls = extract_tool_calls(last, tool_names, settings.mode) # return list[tuples]
        # filter the tool calls that need approval
        gated = [(n, a, i) for n , a, i in calls if _needs_approval(n, a)] 
        if not gated:
            return {"approval_declined": False}

        # inputs are blocking, so we need to run in an executor
        loop = asyncio.get_event_loop()
        # loop through the gated tool calls
        for name, args, _ in gated:
            print(f"\n⚠️  Approval needed: {name}({args})")
            choice = await loop.run_in_executor(
                None, input, "Approve? [y/n/a/lways/N/ever/d(iff)]: "
            )
            choice = choice.strip().lower()
            if choice.startswith("d"):
                print(_preview_diff(name, args))
                choice = await loop.run_in_executor(None, input, "Approve? [y/n/a/N]: ").strip().lower()
            if choice.startswith("a"):
                persist_rule(f"{name}:*", "allow")
                continue
            if choice.startswith("n") and choice != "no":  # N = never
                if choice in ("n", "no"):
                    return {"messages": [], "approval_declined": True}
                persist_rule(f"{name}:*", "deny")
                return {"messages": [], "approval_declined": True}
            if choice not in ("y", "yes", "a", "always"):
                return {"messages": [], "approval_declined": True}
        return {"approval_declined": False}

    # 5. Define the Tools Node
    async def tools_node(state:AgentState) -> AgentState:
        # get the last message
        last = state["messages"][-1]
        # extract the last tool calls
        calls = extract_tool_calls(last, tool_names, settings.mode) # returns list[tuples]
        if not calls:
            return {"messages": []}

        results: list[ToolMessage] = []
        for name, args, call_id in calls:
            # check the permission
            perm = check(name, args)
            if perm == "deny":
                results.append(ToolMessage(
                    tool_call_id=call_id or name,
                    content=f"Denied by permission rule for {name}",
                    name=name,
                ))
                continue
            
            # run the pre-tool use hook
            ok, msg = run_hooks("preToolUse", {"tool": name, "args": args})
            if not ok:
                results.append(ToolMessage(tool_call_id=call_id or name, content=f"Hook veto: {msg}", name=name))
                continue

            # run the tool
            t0 = time.time()
            fn = tool_map.get(name)
            try:
                if fn is None:
                    result = f"Unknown tool: {name}"
                elif hasattr(fn, "ainvoke"):
                    result = await fn.ainvoke(args)
                elif hasattr(fn, "invoke"):
                    result = fn.invoke(args)
                else:
                    result = fn(**args)
            except Exception as e:
                result = f"Error: {e}"

            # run the post-tool use hook
            run_hooks("postToolUse", {"tool": name, "args": args, "result": str(result)})
            duration = int((time.time() - t0) * 1000)
            
            # append the event to the session thread
            append_event("thread", {
                "kind": "tool", "tool": name, "args": args,
                "result": str(result)[:2000], "duration_ms": duration,
            })
            results.append(ToolMessage(
                tool_call_id=call_id or name,
                content=str(result),
                name=name,
            ))
        return {"messages": results}

    # 6. Conditional Routes
    def route_after_agent(state:AgentState) -> Literal["tools", "approval_gate", END]:
        last = state["messages"][-1]
        calls = extract_tool_calls(last, tool_names, settings.mode)
        # 3 routes:
        # 1. no calls: END
        # 2. some calls need approval: approval_gate
        # 3. all calls are allowed: tools
        if not calls:
            return END
        if any(_needs_approval(n, a) for n, a, _ in calls):
            return "approval_gate"
        return "tools"

    def route_after_approval(state:AgentState) -> Literal["tools", "agent"]:
        # if the approval is declined, return to the agent
        # otherwise, return to the tools
        if state["approval_declined"]:
            return "agent"
        return "tools"

    # 7. Build Graph
    g = StateGraph(AgentState)
    g.add_node("agent", agent_node)
    g.add_node("approval_gate", approval_gate)
    g.add_node("tools", tools_node)

    g.add_edge(START, "agent")
    g.add_conditional_edges("agent", route_after_agent)
    g.add_conditional_edges("approval_gate", route_after_approval)
    g.add_edge("tools", "agent")
    
    return g.compile(checkpointer=MemorySaver())