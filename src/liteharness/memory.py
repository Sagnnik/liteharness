"""
There are 3 main memory files:
- NESS.md: project memory (durable, human-managed) (L1)
- runtime/sessions/mem_<thread_id>.md: episodic per-session memory (session-durable, agent-managed) (L3)
- USER.md: cross-repo identity / preferences (L1, human-authored only; CLI stores globally)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from liteharness.options import MemoryConfig

MAX_NESS_CHARS = 20_000
MAX_NESS_INCLUDE_CHARS = 40_000
_NESS_INCLUDE_RE = re.compile(r"^@(\S+)\s*$")


@runtime_checkable
class MemoryBackend(Protocol):
    """Pluggable project / user / session memory backend.

    Inject via ``AgentSpec.memory_store=...``. When omitted,
    :class:`MemoryStore` is constructed from :class:`MemoryConfig`.
    """

    @property
    def disabled(self) -> bool: ...

    def load_project(self) -> str: ...

    def append_project(self, text: str) -> str: ...

    def write_project(self, text: str, overwrite: bool = False) -> str: ...

    def load_user(self) -> str: ...

    def append_user(self, text: str) -> str: ...

    def write_user(self, text: str, overwrite: bool = False) -> str: ...

    def load_session(self, thread_id: str) -> str: ...

    def append_session_bullets(self, thread_id: str, bullets: list[str]) -> bool: ...

    def read_session_raw(self, thread_id: str) -> str: ...

    def write_session_raw(self, thread_id: str, text: str) -> None: ...

    def check_health(self) -> str | None: ...


class MemoryStore:
    """Filesystem-backed project/user/session memory at configurable paths."""

    def __init__(
        self,
        config: MemoryConfig,
        ness_dir: Path | None = None,
        *,
        project_root: Path | None = None,
    ) -> None:
        self.cfg = config
        self.ness_dir = ness_dir or Path.home() / ".ness"
        self.project_root = (project_root or Path.cwd()).resolve()
        self.ness_file = config.project_memory or self.ness_dir / "NESS.md"
        self.user_file = config.user_memory or self.ness_dir / "USER.md"
        self.session_dir = (
            config.session_memory_dir
            or self.ness_dir / "runtime" / "sessions"
        )

    @property
    def disabled(self) -> bool:
        return self.cfg.disabled

    def _expand_includes(self, text: str) -> tuple[str, list[Path]]:
        """Inline standalone ``@path`` directives with referenced file contents.

        Resolves relative to the **project root**, rejects escapes, skips
        missing files, guards against include cycles, and caps total inlined
        size. Returns the expanded text and the ordered list of resolved
        files that were inlined.
        """
        collected: list[Path] = []
        seen: set[Path] = set()
        budget = [MAX_NESS_INCLUDE_CHARS]
        root = self.project_root

        def expand(body: str) -> str:
            out = []
            for line in body.splitlines():
                match = _NESS_INCLUDE_RE.match(line)
                if not match:
                    out.append(line)
                    continue

                ref = match.group(1)

                try:
                    cand = (root / ref).resolve()
                    cand.relative_to(root)
                except (ValueError, OSError):
                    out.append(f"# (invalidinclude: {ref})")
                    continue

                if not cand.is_file():
                    out.append(f"# (missing include: {ref})")
                    continue

                if cand in seen:
                    out.append(f"# (skipped circular: {ref})")
                    continue

                seen.add(cand)
                collected.append(cand)

                if budget[0] <= 0:
                    out.append(f"# (budget exceeded: {ref})")
                    continue

                try:
                    content = cand.read_text(encoding="utf-8")
                except OSError:
                    out.append(f"# (unreadable include: {ref})")
                    continue

                if len(content) > budget[0]:
                    content = content[: budget[0]]
                budget[0] -= len(content)

                out.append(expand(content).rstrip("\n"))
            return "\n".join(out)

        return expand(text), collected

    # project memory (NESS.md)
    def load_project(self) -> str:
        """Load project memory from NESS.md, resolving ``@include`` directives."""
        if self.disabled or not self.ness_file.exists():
            return ""

        expanded, _ = self._expand_includes(self.ness_file.read_text(encoding="utf-8"))
        return expanded

    def append_project(self, text: str) -> str:
        """Append text to NESS.md (manual / CLI ``/memory``)."""
        if self.disabled:
            return "disabled"
        return self._append_markdown(self.ness_file, text)

    def write_project(self, text: str, overwrite: bool = False) -> str:
        """Create/overwrite NESS.md (CLI ``/init``)."""
        if self.disabled:
            return "disabled"
        if self.ness_file.exists() and not overwrite:
            return f"Error: {self.ness_file} exists"

        self.ness_file.parent.mkdir(parents=True, exist_ok=True)
        self.ness_file.write_text(text.strip() + "\n", encoding="utf-8")
        return f"Wrote {self.ness_file}"

    # user memory (USER.md)
    def load_user(self) -> str:
        """Load user memory from USER.md."""
        if self.disabled or not self.user_file.exists():
            return ""
        return self.user_file.read_text(encoding="utf-8")

    def append_user(self, text: str) -> str:
        """Append text to USER.md (manual / CLI ``/user``)."""
        if self.disabled:
            return "disabled"
        return self._append_markdown(self.user_file, text)

    def write_user(self, text: str, overwrite: bool = False) -> str:
        """Create/overwrite USER.md."""
        if self.disabled:
            return "disabled"
        if self.user_file.exists() and not overwrite:
            return f"Error: {self.user_file} exists"

        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(text.strip() + "\n", encoding="utf-8")
        return f"Wrote {self.user_file}"

    # session memory (runtime/sessions/mem_<thread_id>.md)
    def _session_path(self, thread_id: str) -> Path:
        return self.session_dir / f"mem_{thread_id}.md"

    def load_session(self, thread_id: str) -> str:
        """Load session memory as bullet lines (``- item``)."""
        if self.disabled or not thread_id:
            return ""
        p = self._session_path(thread_id)
        if not p.exists():
            return ""
        bullets = []
        for ln in p.read_text(encoding="utf-8").splitlines():
            stripped = ln.strip()
            if stripped.startswith("- "):
                bullets.append(stripped[2:].strip())

        return "\n".join(f"- {b}" for b in bullets)

    def append_session_bullets(self, thread_id: str, bullets: list[str]) -> bool:
        """Append bullet points to the per-thread session memory file."""
        if self.disabled or not thread_id:
            return False
        cleaned = [b.strip() for b in bullets if b and b.strip()]

        if not cleaned:
            return False

        p = self._session_path(thread_id)

        existing: list[str] = []
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                s = ln.strip()
                if s.startswith("- "):
                    existing.append(s[2:].strip())
        merged = list(existing)

        for b in cleaned:
            if b not in merged:
                merged.append(b)
        text = "\n".join(f"- {b}" for b in merged)
        if text:
            text += "\n"

        prev = p.read_text(encoding="utf-8") if p.exists() else ""
        if prev == text:
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return True

    def read_session_raw(self, thread_id: str) -> str:
        """Read the full session-memory file for checkpoint snapshotting."""
        if self.disabled or not thread_id:
            return ""
        p = self._session_path(thread_id)
        if not p.exists():
            return ""
        try:
            return p.read_text(encoding="utf-8")
        except OSError:
            return ""

    def write_session_raw(self, thread_id: str, text: str) -> None:
        """Overwrite (or delete) the session-memory file from a checkpoint.

        Empty ``text`` deletes the file when present.
        """
        if self.disabled or not thread_id:
            return
        p = self._session_path(thread_id)
        p.parent.mkdir(parents=True, exist_ok=True)
        if text:
            p.write_text(text, encoding="utf-8")
        elif p.exists():
            p.unlink()

    def check_health(self) -> str | None:
        """Warn if expanded project memory is at or above the size threshold."""
        chars = len(self.load_project())
        if chars > MAX_NESS_CHARS:
            return f"Warning: NESS.md is at {chars} chars (threshold {MAX_NESS_CHARS})."
        return None

    def _append_markdown(self, path: Path, text: str) -> str:
        cleaned = text.strip()
        if not cleaned:
            return f"No changes for {path}"

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(cleaned + "\n")

        return f"Appended to {path}"
