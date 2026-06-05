from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from config import settings
from tools.common import PROJECT_ROOT

"""
Info about Hooks: 
hooks can be used to run commands, ensure policies, add workflows, telemetry, any customization, etc.
common hooks types -> preToolUse, postToolUse (these 2 are currently used in the code), atUserMessage, atSessionStart, etc.
hooks are defined in the .ness/hooks.json file
simple example: 
"postToolUse": [
    {
    "matcher": "write_file|edit_file|multi_edit",
    "command": "python -c \"import sys; print('file formatted')\"",
    "blocking": false
    }
]

connect to telegram bot example:
"postToolUse": [
    {
      "matcher": "*",
      "command": "curl -s -X POST \"https://api.telegram.org/bot$TELEGRAM_BOT_TOKEN/sendMessage\" -d chat_id=$TELEGRAM_CHAT_ID -d text=\"Tool {TOOL} finished\"",
      "blocking": false,
      "timeout": 10
    }
  ]
"""


NESS = Path(settings.ness_dir)
HOOKS_FILE = NESS / "hooks.json"



@dataclass
class Hook:
    event: str 
    matcher: str # regex pattern to match tool name 
    command: str 
    blocking: bool = True # if True the hook will block the execution of the next hook or tool call (will affect the agentic flow)
    timeout: int = 30


def load_hooks() -> list[Hook]:
    # load .ness/hooks.json file and return list ofHook objects
    if not HOOKS_FILE.exists():
        return []
    data = json.loads(HOOKS_FILE.read_text(encoding="utf-8"))
    hooks = []
    for event, items in data.items():
        for item in items:
            hooks.append(
                Hook(
                    event=event,
                    matcher=item.get("matcher", "*"),
                    command=item["command"],
                    blocking=bool(item.get("blocking", True)),
                    timeout=int(item.get("timeout", 30)),
                )
            )
    return hooks


def run_hooks(event: str, payload: dict[str, Any]) -> tuple[bool, str]:
    """Run matching configured hooks. Return (True/False, message)."""
    messages = []
    for hook in load_hooks():
        # check if the hook event matches the event
        if hook.event != event:
            continue

        # check if the hook matcher matches the tool
        tool = str(payload.get("tool", ""))
        if not _match(hook.matcher, tool):
            continue

        # format the command with the payload
        env = _payload_env(payload)
        command = hook.command.format(**{k: str(v) for k, v in env.items()})

        # run the command
        result = subprocess.run(
            command,
            shell=True,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=hook.timeout,
            env={**os.environ, **{k: str(v) for k, v in env.items()}},
        )

        # get the result
        text = (result.stdout + result.stderr).strip()

        # add the result to the messages
        if text:
            messages.append(text)

        # if the hook is blocking and the command failed, return False and the error message
        if hook.blocking and result.returncode != 0:
            return False, text or f"hook failed: {command}"
    
    return True, "\n".join(messages)


def describe_hooks() -> str:
    # return human readable string of hooks
    hooks = load_hooks()
    if not hooks:
        return "No hooks configured"
    return "\n".join(
        f"- {hook.event} matcher={hook.matcher} blocking={hook.blocking} command={hook.command}"
        for hook in hooks
    )


def _match(pattern: str, tool: str) -> bool:
    if pattern == "*":
        return True
    return re.fullmatch(pattern.replace("*", ".*"), tool) is not None


def _payload_env(payload: dict[str, Any]) -> dict[str, Any]:
    # converts payload to environment variables
    # these variables will be replaceed in the command before subprocess.run
    return {
        "TOOL": payload.get("tool", ""),
        "ARGS": json.dumps(payload.get("args", {}), ensure_ascii=False),
        "RESULT": str(payload.get("result", "")),
        "THREAD_ID": payload.get("thread_id", ""),
        "CWD": str(PROJECT_ROOT),
    }
