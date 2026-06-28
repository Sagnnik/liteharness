from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
PROJECT_SKILLS_DIR = ROOT / ".ness" / "skills"

_LAST_ERRORS: list[str] = []

def _split_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if text.startswith("---"):
        # split it in 3 sections by '---'
        parts = text.split("---", 2)
        if len(parts) == 3:
            meta = yaml.safe_load(parts[1]) or {}
            if not isinstance(meta, dict):
                meta = {}
            return meta, parts[2]
    return {}, text

def _load_skill_md(path: Path) -> dict[str, Any] | None:
    try:
        text = path.read_text(encoding="utf-8")
        
        # split the yaml frontmatter from the body if it exists
        meta, body = _split_frontmatter(text)

        # I am using name, descriptions and triggers as skill metadata for now

        # Standard Metadata fields contain: 
        # name, description, allowed tools, 
        # invocation(auto or manual), references, modules, 
        # versioning (version, compatibility, tags, license)
        # TODO: need to be able to parse these as well

        name = str(meta.get("name") or path.parent.name)
        description = str(meta.get("description") or first_heading_or_line(body))
        triggers = _as_list(meta.get("triggers")) or [name.replace("_", " ")]
        references = _as_list(meta.get("references"))
        inline_references, deferred_references = _split_references(path.parent, references)
        return {
            "name": name,
            "description": description,
            "triggers": triggers,
            "tools": _as_list(meta.get("tools")),
            "references": references,
            "inline_references": inline_references,
            "deferred_references": deferred_references,
            "body": body.strip(),
            "source": str(path),
            "format": "skill.md",
        }
    except Exception as exc:
        _LAST_ERRORS.append(f"{path}: {exc}")
        return None


def load_skills() -> dict[str, dict[str, Any]]:
    """Load SKILL.md skills first, then legacy YAML skills."""
    global _LAST_ERRORS
    _LAST_ERRORS = []
    skills: dict[str, dict[str, Any]] = {}

    for skill_file in sorted(PROJECT_SKILLS_DIR.glob("*/SKILL.md")):
        skill = _load_skill_md(skill_file)
        if skill:
            skills.setdefault(skill["name"], skill)

    return skills


def load_skill_errors() -> list[str]:
    return list(_LAST_ERRORS)


def select_skills(user_input: str, skills: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Select skills whose frontmatter triggers appear in the user message."""
    text = user_input.lower()
    matched: list[dict[str, Any]] = []
    for skill in skills.values():
        triggers = [str(t).lower() for t in skill.get("triggers", [])]
        if any(trigger and trigger in text for trigger in triggers):
            matched.append(skill)
    return matched


def select_sticky_skills(
    user_input: str,
    skills: dict[str, dict[str, Any]],
    sticky_names: set[str],
    explicit_names: list[str] | set[str] | None = None,
) -> list[dict[str, Any]]:
    """Load full skill bodies into L2 on trigger match or `/skill <name>`.

    the agent can read `.ness/skills/<name>/SKILL.md` via read_file and keep it in conversation.
    Once a skill is sticky it stays loaded for the session."""
    for skill in select_skills(user_input, skills):
        sticky_names.add(str(skill.get("name", "")))
    for name in explicit_names or []:
        if name in skills:
            sticky_names.add(name)
    return [skills[name] for name in sorted(sticky_names) if name in skills]


def render_skill_catalog(
    skills: dict[str, dict[str, Any]], relevant: set[str] | None = None
) -> str:
    """Render an always-on one-line catalog of every available skill for L1.

    Optionally marks skills whose triggers/name matched the current request so the
    model knows which to consider loading. Full instructions stay deferred."""
    if not skills:
        return ""
    relevant = relevant or set()
    lines = [
        "Skill catalog (one-line summaries; full instructions load on trigger match or /skill <name> user request):"
    ]
    for name in sorted(skills):
        skill = skills[name]
        description = str(skill.get("description", "")).strip().splitlines()
        summary = description[0].strip() if description else "" 
        marker = " [relevant to this request]" if name in relevant else ""
        if summary:
            entry = f"- {name}: {summary}"
        else:
            entry = f"- {name}"
        lines.append(entry + marker)
    return "\n".join(lines)


def inject_skills(base_prompt: str, skills: list[dict[str, Any]]) -> str:
    """Compatibility helper for older callers."""
    if not skills:
        return base_prompt
    from context import render_active_skills

    return base_prompt + "\n\n" + render_active_skills(skills)

def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(value)]


def _split_references(skill_dir: Path, references: list[str], inline_line_limit: int = 20) -> tuple[list[dict[str, str]], list[str]]:
    inline: list[dict[str, str]] = []
    deferred: list[str] = []
    for reference in references:
        path = Path(reference)
        resolved = path if path.is_absolute() else skill_dir / reference
        display = _display_path(resolved)
        if not resolved.exists() or not resolved.is_file():
            deferred.append(display)
            continue
        try:
            lines = resolved.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            _LAST_ERRORS.append(f"{resolved}: {exc}")
            deferred.append(display)
            continue
        if len(lines) <= inline_line_limit:
            inline.append({"path": display, "content": "\n".join(lines)})
        else:
            deferred.append(display)
    return inline, deferred


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def first_heading_or_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped
    return ""
