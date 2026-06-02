from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


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


def build_foundation(mode: str, tools: Iterable[Any]) -> str:
    """Build L1: stable NESS identity, universal rules, output format, and tool catalog."""
    catalog = _render_tool_catalog(tools)
    tool_calling = _xml_tool_calling(tools) if mode == "xml" else _native_tool_calling()
    return f"""You are NESS, an expert software engineer working inside the user's repository.

Universal rules:
- Protect secrets, keys, tokens, credentials, and private data. Never reveal or persist them.
- Read before editing. Use search and small file reads before broad changes.
- Prefer edit_file, multi_edit, and apply_patch for existing files; use write_file only for new files or complete replacement.
- Adapt to permission denials and hook vetoes. Do not retry the same denied operation blindly.
- Keep changes scoped to the user's request and the surrounding code's existing patterns.
- Use todo_write for multi-step implementation work when it helps track execution.
- Final answers are concise: what changed, what was verified, and any unresolved gap.

Mode and cache notes:
- Normal mode can use the full active tool set.
- Plan mode is read-only and should produce an actionable plan without modifying files.

Tool catalog:
{catalog}

{tool_calling}

Output format:
- During work, state concrete actions and discoveries briefly.
- When complete, summarize changed files and verification.
- If blocked, state the blocker and the next concrete input needed."""


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
    task_state: str = "",
    reflection_nudge: str = "",
) -> str:
    """Build L3 working state appended to the current user message.
    Includes the mode, todos, task state, and reflection nudge.
    """
    mode = (agent_mode or "normal").lower()

    # plan mode instructions else normal mode instructions
    if mode == "plan":
        mode_block = "MODE: PLAN\n- Use read-only inspection tools only.\n- Produce or refine a plan; do not modify files."
    else:
        mode_block = "MODE: NORMAL\n- Execute the requested work with the active tool set.\n- Keep durable decisions in NESS.md and volatile task state in STATE.md when appropriate."
    parts = [mode_block]

    # add the todos
    parts.append("TODOS\n" + (todos.strip() or "No todos"))

    # add the task state
    parts.append("STATE.md\n" + (task_state.strip() or "No volatile task state"))

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


def build_plan_prompt(user_input: str) -> str:
    return PLAN_PROMPT.format(user_input=user_input)


def build_compaction_prompt(messages: str) -> str:
    return COMPACTION_PROMPT.format(messages=messages)


def build_init_memory_prompt(project_context: str) -> str:
    return INIT_MEMORY_PROMPT.format(project_context=project_context)


def build_thread_summary_prompt(events: str) -> str:
    return THREAD_SUMMARY_PROMPT.format(events=events)


def build_subagent_prompt(agent_name: str, agent_body: str, parent_context: str = "") -> str:
    return SUBAGENT_PROMPT.format(
        agent_name=agent_name,
        agent_body=agent_body.strip(),
        parent_context=parent_context.strip(),
    )


def _native_tool_calling() -> str:
    return "Tool calling: use native tool calls. The native schemas are authoritative."


def _xml_tool_calling(tools: Iterable[Any]) -> str:
    return f"""Tool calling: use XML tool calls exactly in the forms below. You may call multiple tools in one assistant message.

{build_xml_tool_prompt(tools)}"""



# TODO: Need to be made more thorough and detailed
# ---- Instructions ----
PLAN_PROMPT = """Analyze the user request. If it is simple, reply exactly:
NO_PLAN_NEEDED

If it needs multiple steps, reply with a numbered implementation plan of at most 8 steps.

User request:
{user_input}
"""

STEP_PROMPT = "Execute this step only:\n\n{step}"

COMPACTION_PROMPT = """Summarize the earlier conversation for continued coding work.
Preserve decisions, files touched, tool results, unresolved errors, and next steps.

Messages:
{messages}
"""

INIT_MEMORY_PROMPT = """Create a concise .ness/NESS.md project memory file from this context.
Include project purpose, commands, architecture notes, conventions, and gotchas.

Project context:
{project_context}
"""

THREAD_SUMMARY_PROMPT = """Summarize this LiteHarness thread in one short paragraph for an index.

Events:
{events}
"""

SUBAGENT_PROMPT = """You are the {agent_name} LiteHarness subagent.

Parent context:
{parent_context}

Subagent instructions:
{agent_body}

Return a concise result with files inspected or changed, verification, and blockers."""


def get_system_prompt(mode: str) -> str:
    """Compatibility helper for older callers."""
    return build_system_prompt(mode, [], [], "")
