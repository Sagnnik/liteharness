"""Tool-use hooks: JSON file and/or in-memory callables.

Supported events (executed from the graph ``tools`` node):

- ``preToolUse`` — before a tool runs; blocking failure vetoes the call
- ``postToolUse`` — after a tool runs; stdout/messages are prepended to the
  result (ok flag is informational)

Hooks may be shell commands (``.ness/hooks.json``) or Python callables
registered via :meth:`HookRunner.register` / ``AgentSpec.hooks``.

This module is **not** the same as:

- ``approval_handler`` / ``question_handler`` (interactive gates on the agent)
- ``on_plan_turn`` / ``on_interrupt`` (per-Session coding callbacks)

JSON example::

    {
      "postToolUse": [
        {
          "matcher": "write|edit",
          "command": "python -c \\"import sys; print('file formatted')\\"",
          "blocking": false
        }
      ]
    }
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

HookHandler = Callable[[dict[str, Any]], tuple[bool, str]]


@dataclass
class Hook:
    """One tool-use hook definition.

    Provide either ``command`` (shell) or ``handler`` (callable), not both
    required at once — at run time ``handler`` wins when set.
    """

    event: str  # preToolUse | postToolUse
    matcher: str = "*"
    command: str | None = None
    handler: HookHandler | None = None
    blocking: bool = True
    timeout: int = 30


class HookRunner:
    """Run hooks for a given event and payload.

    Hooks are loaded from an optional JSON file and/or an in-memory list
    seeded at construction / via :meth:`register`.
    """

    def __init__(
        self,
        hooks_file: Path | None = None,
        *,
        project_root: Path = Path.cwd(),
        hooks: Sequence[Hook] | None = None,
    ) -> None:
        """Configure the runner.

        Args:
            hooks_file: Path to the JSON file defining shell hooks.
                        ``None`` means no file is loaded.
            project_root: Working directory for hook subprocesses.
            hooks: Optional in-memory hooks seeded at construction.
        """
        self.hooks_file = hooks_file
        self.project_root = project_root
        self._registered: list[Hook] = list(hooks or [])

    def register(self, hook: Hook) -> None:
        """Append an in-memory hook (callable or shell)."""
        self._registered.append(hook)

    def clear_registered(self) -> None:
        """Remove all in-memory (non-file) hooks."""
        self._registered.clear()

    def load(self) -> list[Hook]:
        """Return file hooks followed by registered in-memory hooks."""
        return self._load_file_hooks() + list(self._registered)

    def _load_file_hooks(self) -> list[Hook]:
        if not self.hooks_file or not self.hooks_file.exists():
            return []

        try:
            data = json.loads(self.hooks_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return []

        hooks: list[Hook] = []
        for event, items in data.items():
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                command = item.get("command")
                if not command:
                    continue
                hooks.append(
                    Hook(
                        event=event,
                        matcher=item.get("matcher", "*"),
                        command=command,
                        blocking=bool(item.get("blocking", True)),
                        timeout=int(item.get("timeout", 30)),
                    )
                )
        return hooks

    def run(self, event: str, payload: dict[str, Any]) -> tuple[bool, str]:
        """Execute all hooks matching *event* and the ``"tool"`` key in *payload*.

        Returns ``(ok, messages)`` where *ok* is ``False`` if a blocking
        hook fails, and *messages* is the concatenated output of matching hooks.
        """
        messages: list[str] = []
        all_ok = True
        for hook in self.load():
            if hook.event != event:
                continue

            tool = str(payload.get("tool", ""))
            if not self._match(hook.matcher, tool):
                continue

            if hook.handler is not None:
                ok, text = self._run_handler(hook, payload)
            elif hook.command:
                ok, text = self._run_command(hook, event, payload)
            else:
                continue

            if text:
                messages.append(text)
            if not ok:
                if event == "preToolUse" and hook.blocking:
                    return False, text or "hook failed"
                if hook.blocking:
                    all_ok = False

        return all_ok, "\n".join(messages)

    def _run_handler(self, hook: Hook, payload: dict[str, Any]) -> tuple[bool, str]:
        try:
            ok, text = hook.handler(payload)  # type: ignore[misc]
            return bool(ok), str(text or "")
        except Exception as exc:
            if hook.blocking:
                return False, f"hook failed: {exc}"
            return True, f"hook failed: {exc}"

    def _run_command(
        self, hook: Hook, event: str, payload: dict[str, Any]
    ) -> tuple[bool, str]:
        env = self._payload_env(payload)
        try:
            command = hook.command.format(**{k: str(v) for k, v in env.items()})  # type: ignore[union-attr]
        except (KeyError, ValueError) as exc:
            if hook.blocking:
                return False, f"hook format failed: {exc}"
            return True, f"hook format failed: {exc}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=hook.timeout,
                env={**os.environ, **{k: str(v) for k, v in env.items()}},
            )
        except subprocess.TimeoutExpired:
            msg = f"hook timed out after {hook.timeout}s: {command}"
            if hook.blocking:
                return False, msg
            return True, msg
        except OSError as exc:
            msg = f"hook failed: {command} ({exc})"
            if hook.blocking:
                return False, msg
            return True, msg

        text = (result.stdout + result.stderr).strip()
        if hook.blocking and result.returncode != 0:
            return False, text or f"hook failed: {command}"
        return True, text

    def _match(self, pattern: str, tool: str) -> bool:
        if pattern == "*":
            return True
        return re.fullmatch(pattern.replace("*", ".*"), tool) is not None

    def _payload_env(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "TOOL": payload.get("tool", ""),
            "ARGS": json.dumps(payload.get("args", {}), ensure_ascii=False),
            "RESULT": str(payload.get("result", "")),
            "THREAD_ID": payload.get("thread_id", ""),
            "CWD": str(self.project_root),
        }

    def describe(self) -> str:
        """Return a human-readable summary of all configured hooks."""
        hooks = self.load()
        if not hooks:
            return "No hooks configured"
        lines = []
        for h in hooks:
            kind = "handler" if h.handler is not None else f"command={h.command}"
            lines.append(
                f"- {h.event} matcher={h.matcher} blocking={h.blocking} {kind}"
            )
        return "\n".join(lines)
