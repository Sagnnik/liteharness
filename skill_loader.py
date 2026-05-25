import yaml
from pathlib import Path

SKILLS_DIR = Path(__file__).parent / "skills"

def load_skills():
    """load all .yaml files from skill/ directory"""
    skills = {}
    if not SKILLS_DIR.exists():
        return skills
    
    for f in SKILLS_DIR.glob("*.yml"):
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
    for name, skill in skills.items():
        triggers = skill.get("triggers", [])
        if any(t.lower() in user_input.lower() for t in triggers):
            matched.append(skill)
    return matched

def inject_skills(base_prompt: str, skills: list[dict]) -> str:
    """Append Active skill instructions to the system prompt"""
    if not skills:
        return base_prompt

    blocks = []
    for s in skills:
        constraints = "\n".join(f"- {c}" for c in s.get("constraints", []))
        workflow = "\n".join(
            f"{i+1}. {step}" for i, step in enumerate(s.get("workflow", []))
        )
        examples = ""
        for ex in s.get("examples", []):
            examples += f"\n### {ex.get('name', 'Example')}\n{ex.get('code', '')}\n"

        block = f"""
=== SKILL: {s['name']} ===
{s.get('description', '')}

When to use: {', '.join(s.get('triggers', []))}

Constraints:
{constraints}

Workflow:
{workflow}
{examples}
"""
        blocks.append(block)
    return base_prompt + "\n\nACTIVE SKILLS:\n" + "\n".join(blocks)