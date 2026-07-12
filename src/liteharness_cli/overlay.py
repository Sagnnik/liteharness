from __future__ import annotations

from liteharness.context.overlay import OverlayContext, OverlayProvider
from liteharness.graph.state import AgentState


def render_todos(todos: list[dict] | None) -> str:
    if not todos:
        return ""
    active = [t for t in todos if t.get("status") != "completed"]
    if not active:
        return ""
    return "\n".join(
        f"- [{t.get('status', 'pending')}] {t.get('id', '')}: {t.get('content', '')}"
        for t in active
    )


class CodingOverlay(OverlayProvider):
    def __init__(
        self,
        *,
        plans_dir: str = ".ness/plans/",
        plan_mode_template: str | None = None,
        act_mode_template: str | None = None,
    ) -> None:
        """Initialize the coding overlay."""
        self.plans_dir = plans_dir
        self.plan_mode_template = plan_mode_template
        self.act_mode_template = act_mode_template

    def sections(self, state: AgentState, ctx: OverlayContext) -> dict[str, str]:
        """
        Render the sections for the coding overlay.

        Returns a dict mapping section name to rendered text, in insertion order:
        - mode_switch (one-shot, on the first act turn after a plan->act toggle)
        - plan_mode (plan mode only; <plan-mode> block with plan instructions)
        - git (git branch + git status --porcelain)
        - compaction
        - todos (only when there are non-completed items)
        - session_memory (distilled episodic bullets for this thread)
        """
        sections: dict[str, str] = {}
        mode = (ctx.agent_mode or "act").lower()

        # /skill request hint (one-shot, cleared after first turn)
        req = list(ctx.activate_skills or [])
        if req:
            names = ", ".join(f'"{n}"' for n in req)
            sections["skill_request"] = (
                "SKILL REQUEST\n"
                f"The user intended to use these skills: {names}. "
                f"Load each with the skill_view tool. "
                f"Call skill_view(name=<skill-name>) to load the full skill content."
            )

        if ctx.mode_switch.strip():
            sections["mode_switch"] = self.act_mode_template or ""

        if mode == "plan":
            tmpl = self.plan_mode_template or ""
            sections["plan_mode"] = f'<plan-mode path="{self.plans_dir}">\n{tmpl}\n</plan-mode>'

        if ctx.git_snapshot.strip():
            sections["git"] = "GIT\n" + ctx.git_snapshot.strip()

        if ctx.compaction_note.strip():
            sections["compaction"] = "COMPACTION\n" + ctx.compaction_note.strip()

        todos = render_todos(ctx.todos)
        if todos.strip():
            sections["todos"] = "TODOS\n" + todos.strip()

        if ctx.session_memory.strip():
            sections["session_memory"] = "SESSION MEMORY\n" + ctx.session_memory.strip()

        loaded = ctx.loaded_skills or []
        if loaded:
            lines = ["LOADED SKILLS"]
            for s in loaded:
                n = s.get("name", "")
                d = s.get("description", "")
                p = s.get("path", "")
                lines.append(f"- {n}: {d}: {p}".rstrip(": "))
            sections["loaded_skills"] = "\n".join(lines)

        return sections
