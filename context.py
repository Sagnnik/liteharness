from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path
from typing import Any

from langchain_core.messages import BaseMessage

INSTRUCTIONS_DIR = Path(__file__).resolve().parent / "instructions"
DEFAULT_PERSONA_ID = "default"
DEFAULT_PERSONA = "You are an expert software engineer working inside the user's repository."


def normalize_agent_mode(mode: str | None) -> str:
    m = (mode or "act").lower()
    return "act" if m == "normal" else m


@lru_cache(maxsize=None)
def load_instruction(stem: str) -> str:
    path = INSTRUCTIONS_DIR / f"{stem}_instructions.md"
    return path.read_text(encoding="utf-8").strip()


def build_system_prompt(
    tools: Iterable[Any],
    active_skills: Iterable[Mapping[str, Any]] | None = None,
    project_context: str = "",
    git_available: bool | None = None,
) -> str:
    """Compatibility wrapper for callers that still expect one prompt string."""
    foundation = "\n\n".join(
        [build_l0(tools), build_l1(DEFAULT_PERSONA, tools)]
    ).strip()
    sections = [
        foundation,
        build_project_context_block(project_context, active_skills or [], git_available=git_available),
    ]
    return "\n\n".join(sections).strip()


def build_l0(tools: Iterable[Any] | None = None) -> str:
    """Build L0: stable harness identity, universal rules, and tool protocol."""
    return load_instruction("l0_harness")


def build_l1(
    persona: str | Mapping[str, Any] | None,
    tools: Iterable[Any],
    user_memory: str = "",
    ness_memory: str = "",
    skill_catalog: str = "",
) -> str:
    """Build L1: profile/persona details, stable tool catalog, skill catalog, USER.md, and NESS.md."""
    catalog = _render_tool_catalog(tools)
    persona_text = _persona_text(persona)
    user_section = _user_memory_section(user_memory)
    ness_section = _ness_memory_section(ness_memory)
    return load_instruction("l1_profile").format(
        persona=persona_text,
        catalog=catalog,
        skill_catalog=skill_catalog.strip(),
        user_section=user_section,
        ness_section=ness_section,
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


_MCP_DESC_MAX_CHARS = 100


def _render_tool_catalog(tools: Iterable[Any]) -> str:
    from tools import catalog_groups_for_render

    names = {getattr(tool, "name", "") for tool in tools if getattr(tool, "name", "")}
    groups = [(label, names & tier_names) for label, tier_names in catalog_groups_for_render()]

    lines = []
    for label, group in groups:
        if group:
            lines.append(f"- {label}: {', '.join(sorted(group))}")

    ungrouped = sorted(name for name in names if not any(name in group for _, group in groups))
    if ungrouped:
        lines.append(f"- Other active tools: {', '.join(ungrouped)}")

    deferred = _render_deferred_mcp_servers()
    if deferred:
        lines.append(deferred)
    return "\n".join(lines) if lines else "- No tools registered"


def _render_deferred_mcp_servers() -> str:
    """List MCP servers whose tools are not yet loaded, so the model knows what it
    can discover via search_tools. Names are cheap; full schemas stay deferred."""
    from tools import ACTIVE_MCP_TOOLS, mcp_catalog

    catalog = mcp_catalog()
    if not catalog:
        return ""

    server_lines: list[str] = []
    for server in sorted(catalog):
        info = catalog[server]
        deferred_count = sum(
            1 for entry in info.get("tools", []) if entry.get("name") not in ACTIVE_MCP_TOOLS
        )
        if deferred_count == 0:
            continue
        desc = str(info.get("description") or "").strip().replace("\n", " ")
        if not desc:
            sample = [
                str(entry.get("tool") or "")
                for entry in info.get("tools", [])
                if entry.get("name") not in ACTIVE_MCP_TOOLS
            ][:4]
            desc = ", ".join(t for t in sample if t)
        if len(desc) > _MCP_DESC_MAX_CHARS:
            desc = desc[:_MCP_DESC_MAX_CHARS].rstrip() + "..."
        suffix = f": {desc}" if desc else ""
        server_lines.append(f"  - mcp__{server}__* ({deferred_count} tool(s)){suffix}")

    if not server_lines:
        return ""
    header = "- Available MCP servers (use search_tools to find, add_tools to load):"
    return "\n".join([header, *server_lines])


def render_todos(todos: list[dict] | None) -> str:
    # used by the agent: - [status] <id>: <content>
    if not todos:
        return "No todos"
    return "\n".join(
        f"- [{todo.get('status', 'pending')}] {todo.get('id', '')}: {todo.get('content', '')}"
        for todo in todos
    )


def build_working_state_overlay(
    agent_mode: str,
    todos: str = "",
    session_memory: str = "",
    git_snapshot: str = "",
    compaction_note: str = "",
) -> str:
    """
    Build L3 working state. The agent wraps this in <system-reminder> tags and sends it as a
    dedicated ephemeral HumanMessage at the tail of the message list (never persisted to state).
    Currently it has:
    - GIT SNAPSHOT (git branch + git status --porcelain)
    - COMPACTION
    - TODOS
    - SESSION MEMORY (distilled episodic bullets for this thread)
    """
    mode = normalize_agent_mode(agent_mode)

    if mode == "plan":
        from config import settings

        plan_path = f"{settings.ness_dir.rstrip('/')}/plans/"
        mode_block = (
            f'<plan-mode path="{plan_path}">\n'
            + load_instruction("plan_mode")
            + "\n</plan-mode>"
        )
    else:
        mode_block = load_instruction("act_mode")
    parts = [mode_block]

    if git_snapshot.strip():
        parts.append("GIT\n" + git_snapshot.strip())

    if compaction_note.strip():
        parts.append("COMPACTION\n" + compaction_note.strip())

    parts.append("TODOS\n" + (todos.strip() or "No todos"))

    if session_memory.strip():
        parts.append("SESSION MEMORY\n" + session_memory.strip())
    return "\n\n".join(parts)


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


def _persona_text(persona: str | Mapping[str, Any] | None) -> str:
    if persona is None:
        return DEFAULT_PERSONA
    if isinstance(persona, Mapping):
        return str(persona.get("text") or persona.get("persona") or DEFAULT_PERSONA).strip()
    return str(persona).strip() or DEFAULT_PERSONA


def _user_memory_section(user_memory: str) -> str:
    if not user_memory.strip():
        return ""
    return (
        "User preferences (cross-repo, authored by the user; honor unless they "
        f"conflict with an explicit request in the current turn):\n{user_memory.strip()}"
    )


def _ness_memory_section(ness_memory: str) -> str:
    if not ness_memory.strip():
        return ""
    return (
        "Project conventions (.ness/NESS.md, which may inline @AGENTS.md / @CLAUDE.md "
        "includes; human-authored, stable; honor unless the current turn explicitly "
        f"overrides):\n{ness_memory.strip()}"
    )


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
    current_session_bullets: str = "",
    todos: str = "",
) -> str:
    return load_instruction("reflection").format(
        thread_id=thread_id,
        user_message_count=user_message_count,
        messages=_messages_to_text(messages),
        current_session_bullets=current_session_bullets.strip() or "(none yet)",
        todos=todos.strip() or "No todos",
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
