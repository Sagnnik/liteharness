from __future__ import annotations

import fnmatch
import json
import os
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse, urlunparse

from config import settings

Decision = Literal["allow", "deny", "ask"]
SHELL_TOOL = "shell"
SHELL_COMMAND_ACTIONS = {"run", "start"}
WEBFETCH_TOOL = "webfetch"

PROJECT_ROOT = Path(os.getcwd()).resolve()


def validate_path(path: str) -> str:
    """Return an absolute path if it is inside the project root."""
    try:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        resolved = candidate.resolve()
        if not resolved.is_relative_to(PROJECT_ROOT):
            raise PermissionError(f"{path} is outside {PROJECT_ROOT}")
        return str(resolved)
    except Exception as exc:
        raise ValueError(f"Invalid path: {path} ({exc})") from exc


def relative_to_root(path: str) -> str:
    """Validate a path and return it relative to the project root."""
    return str(Path(validate_path(path)).relative_to(PROJECT_ROOT))

NESS = Path(settings.ness_dir)
PERMS_FILE = NESS / "permissions.json"

_session_rules: dict[str, list[str]] = {"allow": [], "deny": []}

DEFAULT_RULES = {
    "allow": [
        "read:*",
        "grep:*",
        "glob:*",
        "edit:*",
        "write:*",
        "delete:*",
        "web_search:*",
        "shell:jobs:*",
        "shell:read:*",
        "shell:run:pwd",
        "shell:run:ls*",
        "shell:run:git status*",
        "shell:run:git diff*",
        "shell:run:git log*",
        "shell:run:git show*",
    ],
    "deny": [
        "shell:run:rm*",
        "shell:run:sudo*",
        "shell:run:curl http*",
        "shell:run:wget *",
        "shell:start:rm*",
        "shell:start:sudo*",
        "shell:start:curl http*",
        "shell:start:wget *",
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
    for rule in _session_rules.get("deny", []):
        if _matches(rule, key, tool):
            return "deny", rule
    for rule in rules.get("allow", []):
        if _matches(rule, key, tool):
            return "allow", rule
    for rule in _session_rules.get("allow", []):
        if _matches(rule, key, tool):
            return "allow", rule
    for rule in rules.get("ask", []):
        if _matches(rule, key, tool):
            return "ask", rule
    return "ask", None


def pattern_key(tool: str, args: dict) -> str:
    if tool == SHELL_TOOL:
        action = _shell_action(args)
        if action in SHELL_COMMAND_ACTIONS:
            return f"{SHELL_TOOL}:{action}:{args.get('command', '')}"
        parts = [f"{key}={args[key]}" for key in sorted(args) if key != "action"]
        detail = ",".join(parts) if parts else "*"
        return f"{SHELL_TOOL}:{action}:{detail}"
    if tool == WEBFETCH_TOOL:
        return f"{WEBFETCH_TOOL}:url={_normalize_permission_url(str(args.get('url') or ''))}"
    if not args:
        return tool
    parts = [f"{key}={args[key]}" for key in sorted(args)]
    return f"{tool}:{','.join(parts)}"


def default_rule_for(tool: str, args: dict) -> str:
    if tool == WEBFETCH_TOOL:
        return pattern_key(tool, args)
    if tool != SHELL_TOOL:
        return pattern_key(tool, args)
    action = _shell_action(args)
    if action not in SHELL_COMMAND_ACTIONS:
        return pattern_key(tool, args)
    command = args.get("command", "").strip()
    parts = command.split()
    if not parts:
        return f"{SHELL_TOOL}:{action}:"
    if parts[0] == "git" and len(parts) >= 2:
        return f"{SHELL_TOOL}:{action}:git {parts[1]}*"
    if parts[0] == "python" and len(parts) >= 3 and parts[1] == "-m":
        return f"{SHELL_TOOL}:{action}:python -m {parts[2]}*"
    if parts[0] in {"npm", "npx", "pnpm", "yarn"} and len(parts) >= 3 and parts[1] == "run":
        return f"{SHELL_TOOL}:{action}:{parts[0]} run {parts[2]}*"
    return f"{SHELL_TOOL}:{action}:{parts[0]}*"


RuleScope = Literal["always", "session"]


def persist_rule(
    rule: str,
    bucket: Literal["allow", "deny"],
    scope: RuleScope = "always",
) -> None:
    if scope == "session":
        if rule not in _session_rules[bucket]:
            _session_rules[bucket].append(rule)
        return
    rules = _load()
    rules.setdefault(bucket, [])
    if rule not in rules[bucket]:
        rules[bucket].append(rule)
    _save(rules)


def clear_session_rules() -> None:
    _session_rules["allow"].clear()
    _session_rules["deny"].clear()


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


def _has_unquoted_shell_operators(command: str) -> bool:
    """Return True when command contains unquoted shell chaining or substitution."""
    in_single = False
    in_double = False
    i = 0
    length = len(command)
    while i < length:
        char = command[i]
        if in_single:
            if char == "'":
                in_single = False
            i += 1
            continue
        if in_double:
            if char == "\\" and i + 1 < length:
                i += 2
                continue
            if char == '"':
                in_double = False
            i += 1
            continue
        if char == "'":
            in_single = True
            i += 1
            continue
        if char == '"':
            in_double = True
            i += 1
            continue
        for operator in ("&&", "||", "$(", "\n"):
            if command.startswith(operator, i):
                return True
        if char in ";|&`<>":
            return True
        i += 1
    return False


def _shell_command_matches(rule: str, command: str) -> bool:
    """Match shell commands safely — reject unquoted shell chaining bypasses."""
    if _has_unquoted_shell_operators(command):
        return False
    return fnmatch.fnmatch(command.strip(), rule.strip())


def _shell_action(args: dict) -> str:
    return str(args.get("action") or "run").strip().lower()


def _normalize_permission_url(url: str) -> str:
    parsed = urlparse(url)
    hostname = parsed.hostname
    if not parsed.scheme or not hostname:
        return url.strip()

    scheme = parsed.scheme.lower()
    host = hostname.lower().rstrip(".")
    netloc = host
    if ":" in host and not host.startswith("["):
        netloc = f"[{host}]"
    try:
        port = parsed.port
    except ValueError:
        return url.strip()
    if port and not _is_default_port(scheme, port):
        netloc = f"{netloc}:{port}"
    return urlunparse((scheme, netloc, parsed.path or "", parsed.params or "", parsed.query or "", ""))


def _is_default_port(scheme: str, port: int) -> bool:
    return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)


def _matches(rule: str, key: str, tool: str) -> bool:
    if rule == "*":
        return True
    if ":" not in rule:
        return fnmatch.fnmatch(tool, rule) or fnmatch.fnmatch(key, rule)
    rule_tool, rule_args = rule.split(":", 1)
    key_tool, _, key_args = key.partition(":")
    if rule_tool != key_tool:
        return False
    if rule_tool == SHELL_TOOL:
        if rule_args == "*":
            key_action, _, key_detail = key_args.partition(":")
            if key_action in SHELL_COMMAND_ACTIONS:
                return _shell_command_matches("*", key_detail)
            return True
        rule_action, _, rule_detail = rule_args.partition(":")
        key_action, _, key_detail = key_args.partition(":")
        if rule_action != "*" and rule_action != key_action:
            return False
        if key_action in SHELL_COMMAND_ACTIONS:
            return _shell_command_matches(rule_detail, key_detail)
        return fnmatch.fnmatch(key_detail, rule_detail)
    if rule_tool == WEBFETCH_TOOL:
        return key_args == rule_args
    return fnmatch.fnmatch(key_args, rule_args)
