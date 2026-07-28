from __future__ import annotations
from pathlib import Path
from typing import Any
import yaml

class SkillLoader:
    def __init__(self, skills_dir: Path | None = None) -> None:
        self.skills_dir = skills_dir
        # Warnings from the most recent ``load`` (unreadable / invalid skill
        # files). Surfaced by the CLI's /skill command; reset on every load.
        self.errors: list[str] = []

    def load(self) -> dict[str, dict[str, Any]]:
        self.errors = []
        if self.skills_dir is None or not self.skills_dir.exists():
            return {}

        skills: dict[str, dict[str, Any]] = {}

        for path in sorted(self.skills_dir.glob("*/SKILL.md")):
            skill = self._load_skill_md(path)
            if skill:
                skills.setdefault(skill["name"], skill)

        return skills

    def render_catalog(self, skills):
        """Render an always-on one-line catalog of every available skill for L1.
        Format:
        - Skill name: Skill description: Source file path
        Full skill content loads via the skill_view tool."""

        if not skills:
            return ""

        lines = ["Skill catalog (one-line summaries; load full instructions via the skill_view tool):"]
        for name in sorted(skills):
            desc = str(skills[name].get("description", "")).strip().splitlines()
            summary = desc[0].strip() if desc else ""
            source = str(skills[name].get("source", ""))
            line = f"- {name}: {summary}: {source}" if summary else f"- {name}: {source}"
            lines.append(line)
        return "\n".join(lines)

    def _split_frontmatter(self, text: str) -> tuple[dict[str, Any], str]:
        if text.startswith("---"):
            parts = text.split("---", 2)
            if len(parts) == 3:
                meta = yaml.safe_load(parts[1]) or {}
                if not isinstance(meta, dict):
                    meta = {}
                return meta, parts[2]
        return {}, text

    def _load_skill_md(self, path: Path) -> dict[str, Any] | None:
        try:
            text = path.read_text(encoding="utf-8")
            meta, body = self._split_frontmatter(text)

            raw_name = meta.get("name")
            if not raw_name or not str(raw_name).strip():
                self.errors.append(f"{path}: missing 'name' frontmatter")
                return None

            raw_description = meta.get("description")
            if not raw_description or not str(raw_description).strip():
                self.errors.append(f"{path}: missing 'description' frontmatter")
                return None

            return {
                "name": str(raw_name).strip(),
                "description": str(raw_description).strip(),
                "license": str(meta.get("license") or "").strip(),
                "compatibility": str(meta.get("compatibility") or "").strip(),
                "metadata": meta.get("metadata") or {},
                "body": body.strip(),
                "source": str(path),
            }
        except Exception as exc:
            self.errors.append(f"{path}: {exc}")
            return None
