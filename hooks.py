from pathlib import Path
from config import settings
from dataclasses import dataclass
import json
import re
import subprocess

NESS = Path(settings.ness_dir)
HOOKS_FILE = NESS / "hooks.json"

@dataclass
class Hook:
    event: str
    matcher: str
    command: str
    blocking: bool = True

def load_hooks() -> list[Hook]:
    if not HOOKS_FILE.exists():
        return []
    data = json.loads(HOOKS_FILE.read_text())
    hooks = []
    for event, items in data.items():
        for h in items:
            hooks.append(Hook(
                event = event,
                matcher = h.get("matcher", "*"),
                command = h['command'],
                blocking = h.get("blocking", True)
            ))

    return hooks

def run_hooks(event:str, payload:dict) -> tuple[bool, str]:
    """Returns (ok, message). ok=False means veto"""
    tool = payload.get("tool", "")
    for h in load_hooks():
        if h.event != event:
            continue
        if not re.match(h.matcher.replace("*", ".*"), tool):
            continue
        env = {**payload, "TOOL": tool, "ARGS": json.dumps(payload.get("args", {}))}
        cmd = h.command.format(**{k: str(v) for k, v in env.items()})
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30)

        if h.blocking and r.returncode != 0:
            return False, r.stderr or r.stdout or "hook vetoed"

    return True, ""


