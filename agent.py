from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from typing import TypedDict, Annotated, Sequence
import operator
from langchain_openrouter import ChatOpenRouter
import os
from dotenv import load_dotenv

load_dotenv()

from tools import (
    read_file,
    write_file,
    apply_diff,
    list_files,
    get_project_context,
    search_files,
    git_snapshot,
    git_commit,
    git_diff,
    run_tests,
)
from parsers import parse_xml_tools

# 1. Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

# 2. Tool Registry
ALL_TOOLS = []

# 3. Prompt
SYSTEM_PROMPT = """You are an expert software engineer. You help users write and modify code. 
You have access to these tools (use XML tags exactly as shown):

<read_file>
    <path>FILE_PATH</path>
</read_file>

<write_file>
    <path>FILE_PATH</path>
    <content>FILE_CONTENT</content>
</write_file>

<apply_diff>
    <path>FILE_PATH</path>
    <old_string>TEXT_TO_FIND</old_string>
    <new_string>TEXT_TO_REPLACE</new_string>
</apply_diff>

<list_files>
    <path>DIR_PATH</path>
</list_files>

<get_project_context>
</get_project_context>

<search_files>
    <path>DIR_PATH</path>
    <query>SEARCH_QUERY</query>
</search_files>

<git_snapshot>
    <message>COMMIT_MESSAGE</message>
</git_snapshot>

<git_commit>
    <message>COMMIT_MESSAGE</message>
</git_commit>

<git_diff>
</git_diff>

<run_tests>
    <test_path>OPTIONAL_PATH</test_path>
</run_tests>

RULES:
- ALWAYS read a file before modifying it
- Wrap EVERY tool call in XML tags as shown above
- You may use multiple tools in one response
- After writing code, run tests to verify
- Use git_snapshot before destructive edits
- After all tools, summarize what you did
- NEVER output raw code outside XML tags
- If a task involves multiple files, explain your plan first"""
# 4. Graph
model = ChatOpenRouter(model="deepseek/deepseek-v4-flash", api_key=os.getenv("OPENROUTER_API_KEY"))
# 4.1 Main Agent Node
async def agent_node(state:AgentState) -> AgentState:
    messages = list(state["messages"])
    # get the last human message
    user_input = next((m.content for m in reversed(messages) if m.type == "human"), "")
    
    #TODO: load skills

    base = SYSTEM_PROMPT
    # get the assistant messages
    has_assitant = any(
        getattr(m, "type", None) in ("ai", "tool", "assistant") for m in messages
    )

    # if no assistant messages then build the project context first
    if not has_assitant:
        ctx = get_project_context()
        sys_msg = SystemMessage(content=f"{base}\n\nProject Context:\n{ctx}")
        messages = [sys_msg] + [m for m in messages if m.type != "system"]

    else:
        messages = [SystemMessage(content=base)] + [m for m in messages if m.type != "system"]

    #TODO: trim messages or summarize for context management

    # invoke llm
    response = await model.ainvoke(messages)

    #TODO: cost tracker

    return {"messages": [response]}


# 4.2 Approval Gate
async def approval_gate(state:AgentState) -> AgentState:
    last_msg = state["messages"][-1]
    calls = parse_xml_tools(last_msg.content)
    
    if not calls:
        return {"messages": []}

    #TODO: check if the tools are in DESTRUCTIVE_TOOLS list

    


# 4.3 Tools Node

# 4.4 Conditional Router

# 5. Workflow
