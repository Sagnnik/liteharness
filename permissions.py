import fnmatch
import json
from pathlib import Path
from typing import Literal

from config import settings

NESS = Path(settings.ness_dir)
PERMS_FILE = NESS / "permissions.json"

DEFAULT_RULES = {
    "allow": [
        "bash:ls*",
        "bash:pwd",
        "bash:cat *",
        "bash:head *",
        "bash:tail *",
        "bash:git status*",
        "bash:git diff*",
        "bash:git log*",
        "bash:git show*",
        "bash:git blame*",
        "grep:*",
        "glob:*",
        "read_file:*",
        "list_files:*",
        "git_status",
        "git_diff:*",
        "git_log:*",
        "git_show:*",
        "git_blame:*",
        "todo_read",
    ],
    "deny": [
        "bash:rm -rf*",
        "bash:sudo*",
        "bash:curl http*",
        "bash:wget *",
    ],
    "ask": ["*"],
}


def _load() -> dict:
    if not PERMS_FILE.exists():
        NESS.mkdir(parents=True, exist_ok=True)
        PERMS_FILE.write_text(json.dumps(DEFAULT_RULES, indent=2))
        return DEFAULT_RULES.copy()
    return json.loads(PERMS_FILE.read_text())


def _save(rules: dict):
    NESS.mkdir(parents=True, exist_ok=True)
    PERMS_FILE.write_text(json.dumps(rules, indent=2))


def _pattern_key(tool: str, args: dict) -> str:
    if tool == "bash":
        return f"bash:{args.get('command', '')}"
    parts = [f"{k}={v}" for k, v in sorted(args.items())]
    return f"{tool}:{','.join(parts)}" if parts else tool


def _matches(rule: str, key: str) -> bool:
    if rule == "*":
        return True
    if ":" in rule:
        tool, pat = rule.split(":", 1)
        if tool != key.split(":", 1)[0]:
            return False
        return fnmatch.fnmatch(key.split(":", 1)[1], pat)
    return fnmatch.fnmatch(key, rule)


def check(tool: str, args: dict) -> Literal["allow", "deny", "ask"]:
    rules = _load()
    key = _pattern_key(tool, args)
    for r in rules.get("deny", []):
        if _matches(r, key) or _matches(r, tool):
            return "deny"
    for r in rules.get("allow", []):
        if _matches(r, key) or _matches(r, tool):
            return "allow"
    return "ask"


def persist_rule(rule: str, bucket: Literal["allow", "deny"]):
    rules = _load()
    if rule not in rules[bucket]:
        rules[bucket].append(rule)
    _save(rules)


def list_rules() -> str:
    return json.dumps(_load(), indent=2)