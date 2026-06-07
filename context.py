from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

INSTRUCTIONS_DIR = Path(__file__).resolve().parent / "instructions"


@lru_cache(maxsize=None)
def load_instruction(stem: str) -> str:
    path = INSTRUCTIONS_DIR / f"{stem}_instructions.md"
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(
    mode: str,
    tools: Iterable[Any],
    active_skills: Iterable[Mapping[str, Any]] | None = None,
    project_context: str = "",
) -> str:
    """Compatibility wrapper for callers that still expect one prompt string."""
    sections = [
        build_foundation(mode, tools),
        build_project_context_block(project_context, active_skills or [], git_available=None),
    ]
    return "\n\n".join(sections).strip()


def build_foundation(mode: str, tools: Iterable[Any], user_memory: str = "") -> str:
    """Build L1: stable NESS identity, universal rules, output format, and tool catalog.

    user_memory holds cross-repo user preferences (USER.md). It is authored by
    the user, so it lives in the most-cached layer and is honored unless it
    conflicts with an explicit request in the current turn.
    """
    catalog = _render_tool_catalog(tools)
    tool_calling = _xml_tool_calling(tools) if mode == "xml" else _native_tool_calling()
    user_section = ""
    if user_memory.strip():
        user_section = (
            "\n\nUser preferences (cross-repo, authored by the user; honor unless they "
            f"conflict with an explicit request in the current turn):\n{user_memory.strip()}"
        )
    return load_instruction("foundation").format(
        user_section=user_section,
        catalog=catalog,
        tool_calling=tool_calling,
    )


def build_project_context_block(
    project_context: str = "",
    active_skills: Iterable[Mapping[str, Any]] | None = None,
    git_available: bool | None = None,
) -> str:
    """Build L2: stable per-thread project context and sticky skill cores."""
    sections = ["PROJECT CONTEXT PREFIX"]

    # check if git is available
    if git_available is not None:
        sections.append(f"Git repository: {'yes' if git_available else 'no'}")

    # add the project context
    if project_context:
        sections.append(project_context.strip())

    # add the active skills
    skills = render_active_skills(active_skills or [])
    if skills:
        sections.append(skills)
    return "\n\n".join(section for section in sections if section).strip()


def _render_tool_catalog(tools: Iterable[Any]) -> str:
    names = {getattr(tool, "name", "") for tool in tools}

    # create the tool groups: (group name, tools in the group)
    groups = [("Small always-on", names & {"todo_read", "todo_write", "add_to_memory"}),
        ("L1 core", names & {"read_file", "write_file", "edit_file", "multi_edit", "apply_patch", "grep", "glob_files", "list_files", "bash", "get_project_context"}),
        ("L2 git read", names & {"git_status", "git_diff", "git_log", "git_show", "git_blame"}),
        ("L3 advanced", names & {"git_snapshot", "git_commit", "git_checkout", "git_branch", "git_stash", "git_worktree_add", "git_worktree_list", "git_worktree_remove", "spawn_subagent"}),
        ("Dynamic MCP", {name for name in names if name.startswith("mcp__")}),
    ]

    # proper rendering: - group name: tool1, tool2, tool3
    lines = []
    for label, group in groups:
        if group:
            lines.append(f"- {label}: {', '.join(sorted(group))}")
    
    # add the ungrouped tools 
    ungrouped = sorted(name for name in names if name and not any(name in group for _, group in groups))
    if ungrouped:
        lines.append(f"- Other active tools: {', '.join(ungrouped)}")
    return "\n".join(lines) if lines else "- No tools registered"


def build_working_state_overlay(
    agent_mode: str,
    todos: str = "",
    reflection_nudge: str = "",
) -> str:
    """Build L3 working state appended to the current user message.
    Includes the mode, todos, and reflection nudge.
    """
    mode = (agent_mode or "normal").lower()

    # plan mode instructions else normal mode instructions
    if mode == "plan":
        mode_block = load_instruction("plan_mode")
    else:
        mode_block = load_instruction("normal_mode")
    parts = [mode_block]

    # add the todos
    parts.append("TODOS\n" + (todos.strip() or "No todos"))

    # add the reflection nudge
    if reflection_nudge:
        parts.append("REFLECTION\n" + reflection_nudge.strip())
    return "\n\n".join(parts)


def build_xml_tool_prompt(tools: Iterable[Any]) -> str:
    """Render XML fallback examples from the current tool registry."""
    blocks = []
    for tool in sorted(tools, key=lambda item: getattr(item, "name", "")):
        name = getattr(tool, "name", "")
        if not name:
            continue
        schema = getattr(tool, "args_schema", None)
        fields = getattr(schema, "model_fields", None)
        fields = list(fields.keys()) if isinstance(fields, dict) else []
        if fields:
            body = "\n".join(f"  <{field}>{field.upper()}</{field}>" for field in fields)
            blocks.append(f"<{name}>\n{body}\n</{name}>")
        else:
            blocks.append(f"<{name}></{name}>")
    return "\n\n".join(blocks)


def render_active_skills(skills: Iterable[Mapping[str, Any]]) -> str:
    """
    Renders the active skills into a string. Caught by triggers in the conversation.
    skills are picked from the .ness/skills folder via the select_sticky_skills() function and are kept in the L2 prompt section.
    
    General SKILL.md format (a lot of them are optional and are not needed for the prompt):
    - name: <name>
    - description: <description>
    - constraints: <constraints> (optional list of constraints)
    - workflow: <workflow> (optional list of steps)
    - body: <body> (after the frontmatter)
    - source: <source> (path to SKILL.md file)
    - inline_references: <inline_references> (pasted in)
    - deferred_references: <deferred_references> (file paths only)
    - triggers: <triggers> (frontmatter)
    - tools: <tools> (frontmatter)
    - versioning: <version>
    - compatibility: <compatibility>
    - tags: <tags>
    - license: <license>
    - author: <author>
    - date: <date>
    - last_updated: <last_updated>
    - status: <status>

    Output format (1 skill block):
    - === SKILL: <name> ===
    - Source: <source>
    - Description: <description>
    - Constraints: <bullet list>
    - Workflow: <numbered steps>
    - Body in markdown format
    - Inline references: small files pasted in
    - Deferred references: <fetch on demand with read_file>
    """
    blocks = []

    # iterate over sorted skills by name (keeps the order stable)
    for skill in sorted(skills, key=lambda item: str(item.get("name", ""))):

        # needed metadata
        name = skill.get("name", "unnamed")
        description = skill.get("description", "")
        body = skill.get("body") or skill.get("instructions") or ""
        source = skill.get("source", "")
        constraints = "\n".join(f"- {c}" for c in skill.get("constraints", []))
        workflow = "\n".join(f"{i + 1}. {step}" for i, step in enumerate(skill.get("workflow", [])))

        # build the skill block
        parts = [f"=== SKILL: {name} ==="]
        if source:
            parts.append(f"Source: {source}")
        if description:
            parts.append(description)
        if constraints:
            parts.append(f"Constraints:\n{constraints}")
        if workflow:
            parts.append(f"Workflow:\n{workflow}")
        if body:
            parts.append(str(body).strip())
        
        # inline references
        inline_refs = skill.get("inline_references", []) or []
        if inline_refs:
            rendered = []
            for ref in inline_refs:
                label = ref.get("path", "reference") if isinstance(ref, Mapping) else "reference"
                content = ref.get("content", "") if isinstance(ref, Mapping) else str(ref)
                rendered.append(f"--- {label} ---\n{str(content).strip()}")
            parts.append("Inlined references:\n" + "\n\n".join(rendered))
        
        # deferred references
        deferred_refs = skill.get("deferred_references", []) or []
        if deferred_refs:
            parts.append(
                "Available references, fetch on demand with read_file:\n"
                + "\n".join(f"- {ref}" for ref in deferred_refs)
            )
        blocks.append("\n".join(parts))
    return "ACTIVE SKILLS\n" + "\n\n".join(blocks) if blocks else ""


def build_compaction_prompt(messages: str) -> str:
    return load_instruction("compaction").format(messages=messages)


def build_init_memory_prompt(project_context: str) -> str:
    return load_instruction("init_memory").format(project_context=project_context)


def build_subagent_prompt(agent_name: str, agent_body: str, parent_context: str = "") -> str:
    return load_instruction("subagent").format(
        agent_name=agent_name,
        agent_body=agent_body.strip(),
        parent_context=parent_context.strip(),
    )


def build_reflection_prompt(
    thread_id: str,
    messages: Iterable[BaseMessage],
    user_message_count: int,
    *,
    max_tool_calls: int = 3,
    max_ness_chars: int = 12_000,
) -> str:
    return load_instruction("reflection").format(
        thread_id=thread_id,
        user_message_count=user_message_count,
        messages=_messages_to_text(messages),
        max_tool_calls=max_tool_calls,
        max_ness_chars=max_ness_chars,
    )


def build_thread_summary_prompt(events: str) -> str:
    return load_instruction("thread_summary").format(events=events)


def _messages_to_text(messages: Iterable[BaseMessage], limit: int = 16) -> str:
    recent = list(messages)[-limit:]
    return "\n\n".join(f"{msg.type}: {_content_text(msg.content)[:1200]}" for msg in recent)


def _content_text(content) -> str:
    if isinstance(content, list):
        return " ".join(str(item.get("text", item)) if isinstance(item, dict) else str(item) for item in content)
    return str(content)


def _native_tool_calling() -> str:
    return load_instruction("native_tool_calling")


def _xml_tool_calling(tools: Iterable[Any]) -> str:
    return load_instruction("xml_tool_calling").format(tool_examples=build_xml_tool_prompt(tools))
