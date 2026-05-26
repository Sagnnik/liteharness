from langchain_core.tools import tool
from config import settings
from pathlib import Path
import yaml

AGENTS_DIR = Path(settings.ness_dir) / "agents"

def _load_agent(name:str) -> dict:
    path = AGENTS_DIR / f"{name}.md"
    if not path.exists():
        raise FileNotFoundError(f"No Agent: {name}")

    text = path.read_text()
    if text.startswith("---"):
        _, fm, body = text.split("---", 2)
        meta = yaml.safe_load(fm)
        meta["prompt"] = body.strip()
        return meta
    return {"prompt": text, "tools": ["read_file", "grep", "glob_files", "list_files"]}

@tool
def spawn_subagent(name:str, prompt:str) -> str:
    """Spawn an isolated subagent. Name maps to .ness/agents/<name>.md"""
    # TODO: Need to implement this!
    meta = _load_agent(name)
    return f"[subagent: {name}] Would run with tools={meta.get('tools')} prompt={prompt[:200]}"