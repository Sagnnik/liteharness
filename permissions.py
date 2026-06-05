from __future__ import annotations

import fnmatch
import json
from pathlib import Path
from typing import Literal

from config import settings

Decision = Literal["allow", "deny", "ask"]

NESS = Path(settings.ness_dir)
PERMS_FILE = NESS / "permissions.json"

DEFAULT_RULES = {
    "allow": [
        "read_file:*",
        "list_files:*",
        "grep:*",
        "glob_files:*",
        "git_status",
        "git_diff:*",
        "git_log:*",
        "git_show:*",
        "git_blame:*",
        "git_worktree_list",
        "todo_read",
        "get_project_context",
        "bash:pwd",
        "bash:ls*",
        "bash:git status*",
        "bash:git diff*",
        "bash:git log*",
        "bash:git show*",
    ],
    "deny": [
        "bash:rm -rf*",
        "bash:sudo*",
        "bash:curl http*",
        "bash:wget *",
    ],
    "ask": ["*"],
}


def check(tool: str, args: dict) -> Decision:
    decision, _ = check_with_rule(tool, args)
    return decision


def check_with_rule(tool: str, args: dict) -> tuple[Decision, str | None]:
    rules = _load()
    key = pattern_key(tool, args)
    for rule in rules.get("deny", []):
        if _matches(rule, key, tool):
            return "deny", rule
    for rule in rules.get("allow", []):
        if _matches(rule, key, tool):
            return "allow", rule
    for rule in rules.get("ask", []):
        if _matches(rule, key, tool):
            return "ask", rule
    return "ask", None


def pattern_key(tool: str, args: dict) -> str:
    if tool == "bash":
        return f"bash:{args.get('command', '')}"
    if not args:
        return tool
    parts = [f"{key}={args[key]}" for key in sorted(args)]
    return f"{tool}:{','.join(parts)}"


def default_rule_for(tool: str, args: dict) -> str:
    return pattern_key(tool, args)


def persist_rule(rule: str, bucket: Literal["allow", "deny"]) -> None:
    rules = _load()
    rules.setdefault(bucket, [])
    if rule not in rules[bucket]:
        rules[bucket].append(rule)
    _save(rules)


def remove_rule(bucket: Literal["allow", "deny"], index: int) -> str:
    rules = _load()
    try:
        removed = rules.get(bucket, []).pop(index)
    except IndexError as exc:
        raise ValueError(f"No {bucket} rule at index {index}") from exc
    _save(rules)
    return removed


def list_rules() -> str:
    return json.dumps(_load(), indent=2)


def _load() -> dict:
    if not PERMS_FILE.exists():
        NESS.mkdir(parents=True, exist_ok=True)
        _save(DEFAULT_RULES)
        return json.loads(json.dumps(DEFAULT_RULES))
    data = json.loads(PERMS_FILE.read_text(encoding="utf-8"))
    for key in ("allow", "deny", "ask"):
        data.setdefault(key, [])
    return data


def _save(rules: dict) -> None:
    NESS.mkdir(parents=True, exist_ok=True)
    PERMS_FILE.write_text(json.dumps(rules, indent=2), encoding="utf-8")


def _matches(rule: str, key: str, tool: str) -> bool:
    if rule == "*":
        return True
    if ":" not in rule:
        return fnmatch.fnmatch(tool, rule) or fnmatch.fnmatch(key, rule)
    rule_tool, rule_args = rule.split(":", 1)
    key_tool, _, key_args = key.partition(":")
    return rule_tool == key_tool and fnmatch.fnmatch(key_args, rule_args)
