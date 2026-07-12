"""
Info about Hooks: 
hooks can be used to run commands, ensure policies, add workflows, telemetry, any customization, etc.
common hooks types -> preToolUse, postToolUse (these 2 are currently used in the code), atUserMessage, atSessionStart, etc.
hooks are defined in the .ness/hooks.json file
simple example: 
"postToolUse": [
    {
    "matcher": "write|edit",
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

from __future__ import annotations
import json, os, re, subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

@dataclass
class Hook:
    event: str  # event name (preToolUse or postToolUse)
    matcher: str  # regex pattern to match the tool name
    command: str  # command to execute
    blocking: bool = True  # non-zero return code is considered a failure
    timeout: int = 60  # timeout in seconds

class HookRunner:
    """Run hooks for a given event and payload.

    Hooks are user-defined commands that execute before or after tool
    invocations (``preToolUse`` / ``postToolUse``).  They are loaded from
    a JSON file whose keys are event names and values are lists of hook
    definitions.
    """

    def __init__(self, hooks_file: Path | None = None, *, project_root: Path = Path.cwd()) -> None:
        """Configure the runner with an optional path to a hooks JSON file.

        Args:
            hooks_file: Path to the JSON file defining hooks.
                        ``None`` (the default) means no hooks are loaded.
            project_root: Working directory for hook subprocesses.
                          Defaults to ``cwd`` at import time.
        """
        self.hooks_file = hooks_file
        self.project_root = project_root

    def load(self) -> list[Hook]:
        """Parse the hooks file and return a list of :class:`Hook` instances.

        Returns an empty list if no ``hooks_file`` was configured, the
        file does not exist, or the file contains invalid JSON.
        """
        if not self.hooks_file or not self.hooks_file.exists(): 
            return []

        try:
            data = json.loads(self.hooks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        hooks = []
        
        for event, items in data.items():
            for item in items:
                command = item.get("command")
                if not command:
                    continue
                hooks.append(
                    Hook(
                        event=event, 
                        matcher=item.get("matcher", "*"),
                        command=command, 
                        blocking=bool(item.get("blocking", True)),
                        timeout=int(item.get("timeout", 30))
                    )
                )
        
        return hooks

    def run(self, event: str, payload: dict[str, Any]) -> tuple[bool, str]:
        """Execute all hooks matching *event* and the ``"tool"`` key in *payload*.

        Each matching hook runs as a ``shell=True`` subprocess.  The
        payload fields ``TOOL``, ``ARGS``, ``RESULT``, ``THREAD_ID``, and
        ``CWD`` are injected as both format variables in the command
        string and environment variables.

        Returns ``(ok, messages)`` where *ok* is ``False`` if a blocking
        hook exits with a non-zero return code, and *messages* is the
        concatenated stdout+stderr of all matching hooks.
        """
        messages = []
        all_ok = True
        for hook in self.load():
            # check if the hook event matches the event
            if hook.event != event: 
                continue
            
            # check if the hook matcher matches the tool
            tool = str(payload.get("tool", ""))
            if not self._match(hook.matcher, tool): 
                continue
            
            # format the command with the payload
            env = self._payload_env(payload)
            command = hook.command.format(**{k: str(v) for k, v in env.items()})
            
            # run the command
            try:
                result = subprocess.run(
                    command, 
                    shell=True, 
                    cwd=self.project_root,
                    capture_output=True, 
                    text=True, 
                    timeout=hook.timeout,
                    env={**os.environ, **{k: str(v) for k, v in env.items()}}
                )
            except subprocess.TimeoutExpired:
                if hook.blocking:
                    if event == "preToolUse":
                        return False, f"hook timed out after {hook.timeout}s: {command}"
                    all_ok = False
                    messages.append(f"hook timed out: {command}")
                continue
            except OSError as exc:
                if hook.blocking:
                    if event == "preToolUse":
                        return False, f"hook failed: {command} ({exc})"
                    all_ok = False
                    messages.append(f"hook failed: {command} ({exc})")
                continue

            # get the result
            text = (result.stdout + result.stderr).strip()

            # add the result to the messages
            if text: 
                messages.append(text)

            # if the hook is blocking and the command failed
            if hook.blocking and result.returncode != 0:
                msg = text or f"hook failed: {command}"
                if event == "preToolUse":
                    return False, msg
                all_ok = False
                messages.append(msg)
        
        return all_ok, "\n".join(messages)

    def _match(self, pattern, tool):
        if pattern == "*": 
            return True
        return re.fullmatch(pattern.replace("*", ".*"), tool) is not None

    def _payload_env(self, payload):
        # converts payload to environment variables
        # these variables will be replaceed in the command before subprocess.run
        return {
            "TOOL": payload.get("tool", ""),
            "ARGS": json.dumps(payload.get("args", {}), ensure_ascii=False),
            "RESULT": str(payload.get("result", "")),
            "THREAD_ID": payload.get("thread_id", ""), "CWD": str(self.project_root)
        }

    def describe(self) -> str:
        """Return a human-readable summary of all configured hooks."""
        hooks = self.load()
        if not hooks: 
            return "No hooks configured"
        return "\n".join(f"- {h.event} matcher={h.matcher} blocking={h.blocking} command={h.command}" for h in hooks)