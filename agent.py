from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import Literal, TypedDict, Annotated, Sequence
import operator
from langchain_openrouter import ChatOpenRouter
import os
import asyncio
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.runnables import Runnable

from utils import trim_messages
load_dotenv()

from tools import (
    read_file,
    write_file,
    apply_diff,
    list_files,
    get_project_context,
    search_files,
    git_snapshot,
    git_diff,
    run_tests,
)
from parsers import format_tool_result, parse_xml_tools
from prompt import SYSTEM_PROMPT
from skill_loader import load_skills, inject_skills, select_skills
from config import settings, cost_tracker


# 1. Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Tool Registry
ALL_TOOLS = [
    read_file, write_file, apply_diff, list_files,
    get_project_context, search_files, git_snapshot, git_diff, run_tests
]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}
DESTRUCTIVE_TOOLS = {"write_file", "apply_diff"}

# 3. Graph
def build_graph(model: ChatOpenRouter) -> Runnable:
    # 3.1 Main Agent Node
    async def agent_node(state:AgentState) -> AgentState:
        messages = list(state["messages"])
        # get the last human message
        user_input = next((m.content for m in reversed(messages) if m.type == "human"), "")
        
        #TODO: load skills - (not sure about if the SKILL.md with references format collides with the .yaml format)
        skills = load_skills()
        active = select_skills(user_input, skills)
        base = SYSTEM_PROMPT
        if active:
            base = inject_skills(base, active)
        
        # get the assistant messages
        has_assitant = any(getattr(m, "type", None) in ("ai", "tool", "assistant") for m in messages)

        # if no assistant messages then build the project context first
        if not has_assitant:
            ctx = get_project_context()
            sys_msg = SystemMessage(content=f"{base}\n\nProject Context:\n{ctx}")
            messages = [sys_msg] + [m for m in messages if m.type != "system"]

        else:
            messages = [SystemMessage(content=base)] + [m for m in messages if m.type != "system"]

        #TODO: trim messages or summarize for context management - (mostly done; not sure about the summarizing part)
        messages = trim_messages(messages, max_chars=settings.max_tokens)

        # invoke llm
        response = await model.ainvoke(messages)

        #TODO: cost tracker
        if response.usage_metadata:
            cost_tracker.add(response.usage_metadata)

        return {"messages": [response]}


    # 3.2 Approval Gate
    async def approval_gate(state:AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        calls = parse_xml_tools(last_msg.content)
        
        if not calls:
            return {"messages": []}

        destructive = [c for c in calls if c[0] in DESTRUCTIVE_TOOLS]
        if not destructive or not settings.enable_approval:
            return {"messages": []}

        print("\n Destructive tools detected:")
        for name, args in destructive:
            print(f" - {name}: {args}")

        loop = asyncio.get_event_loop()
        choice = await loop.run_in_executor(None, input, "Approve? [y/n/diff/show]: ")
        choice = choice.strip().lower()

        #TODO: Complete this!
        if choice in ("diff", "show", "s", "d"):
            for name, args in destructive:
                path = args.get("path", "unknown")
                if os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            print(f"\n--- {path} ---\n{fh.read(600)}\n...")

                    except Exception as e:
                        print(f"Error reading {path}: {e}")

                else:
                    print(f"{path} does not exist yet (will be created)")

            choice = await loop.run_in_executor(None, input, "Approve? [y/n]: ")
            choice = choice.strip().lower()

        if choice in ("y", "yes"):
            return {"messages": []}
        
        return {"messages": [AIMessage(content="User declined the operation. Do not attempt it again without explicit user direction.")]}

    # 3.3 Tools Node
    async def tools_node(state:AgentState) -> AgentState:
        last_msg = state["messages"][-1]
        calls = parse_xml_tools(last_msg.content)

        if not calls: 
            return {"messages": []}

        results = []
        for name, args in calls:
            tool_fn = TOOL_MAP.get(name)
            
            if not tool_fn:
                results.append(format_tool_result(name, f"Error: unknown tool {name}"))
                continue

            try:
                # checking if it is a valid langchain tool with async invoke
                if hasattr(tool_fn, "ainvoke"): 
                    result = await tool_fn.ainvoke(args)
                # checking if it is a valid langchain tool with sync invoke
                elif hasattr(tool_fn, "invoke"): 
                    result = tool_fn.invoke(args)
                # calling the tool with the arguments
                else:
                    result = tool_fn(**args) 
            except Exception as e:
                result = f"Error: {e}"
            
            results.append(format_tool_result(name, result))

        return {"messages": [AIMessage(content="\n".join(results))]}

    # 3.4 Conditional Routers
    def route_after_agent(state:AgentState) -> Literal["approval_gate", "tools", END]:
        last_msg = state["messages"][-1]
        calls = parse_xml_tools(last_msg.content)

        if not calls:
            return END

        needs = any(name in DESTRUCTIVE_TOOLS for name, _ in calls)
        if needs and settings.enable_approval:
            return "approval_gate"
        return "tools"

    def route_after_approval(state:AgentState) -> Literal["tools", "agent"]:
        last_msg = state["messages"][-1]
        if 'declined' in last_msg.content.lower():
            return "agent"
        return "tools"

    # 3.5 Workflow
    workflow = StateGraph(AgentState)

    workflow.add_node("agent", agent_node)
    workflow.add_node("approval_gate", approval_gate)
    workflow.add_node("tools", tools_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_after_agent)
    workflow.add_conditional_edges("approval_gate", route_after_approval)
    workflow.add_edge("tools", "agent")

    checkpointer = MemorySaver()

    return workflow.compile(checkpointer=checkpointer)