import yaml
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

def load_skills():
    """load all .yaml files from skill/ directory"""
    skills = {}
    if not SKILLS_DIR.exists():
        return skills
    
    for f in list(SKILLS_DIR.glob("*.yml")) + list(SKILLS_DIR.glob("*.yaml")):
        try:
            raw = yaml.safe_load(f.read_text())
            if raw and "name" in raw:
                skills[raw["name"]] = raw
        except Exception:
            continue
    return skills

def select_skills(user_input: str, skills: dict) -> list[dict]:
    """Simple keyword matching for now. Later include embedding or LLM"""
    matched = []
    t = user_input.lower()
    for skill in skills.values():
        for tr in skill.get("triggers", []):
            if any(tr.lower() in t):
                matched.append(skill)

    return matched

def inject_skills(base_prompt: str, skills: list[dict]) -> str:
    """Append Active skill instructions to the system prompt"""
    if not skills:
        return base_prompt

    blocks = []
    for s in skills:
        constraints = "\n".join(f"- {c}" for c in s.get("constraints", []))
        workflow = "\n".join(f"{i+1}. {step}" for i, step in enumerate(s.get("workflow", [])))
        examples = "".join(
            f"\n### {ex.get('name', 'Example')}\n{ex.get('code', '')}\n"
            for ex in s.get("examples", [])
        )
        blocks.append(
            f"=== SKILL: {s['name']} ===\n{s.get('description', '')}\n\n"
            f"Constraints:\n{constraints}\n\nWorkflow:\n{workflow}{examples}"
        )
    return base_prompt + "\n\nACTIVE SKILLS:\n" + "\n".join(blocks)