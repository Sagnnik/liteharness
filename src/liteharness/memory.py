"""
There are 3 main memory files:
- NESS.md: project memory (durable, human-managed) (L1)
- sessions/mem_<thread_id>.md: episodic per-session memory (session-durable, agent-managed) (L2)
- USER.md: cross-repo identity / preferences (L1, human-authored only)
"""

from __future__ import annotations
import re 
import json
from pathlib import Path
from liteharness.options import MemoryConfig
from liteharness.permissions import DEFAULT_RULES

NESS_SUBDIRS = (
    "sessions",
    "agents",
    "commands",
    "skills",
    "plans",
    "threads",
    "shells",
)
MAX_NESS_CHARS = 20_000
MAX_NESS_INCLUDE_CHARS = 40_000
_NESS_INCLUDE_RE = re.compile(r"^@(\S+)\s*$")

class MemoryStore:
    """Project/user/session memory at configurable paths"""

    def __init__(self, config: MemoryConfig, ness_dir: Path | None = None) -> None:
        self.cfg = config
        self.ness_dir = ness_dir or Path.home() / ".ness"
        self.ness_file = config.project_memory or self.ness_dir / "NESS.md"
        self.user_file = config.user_memory or self.ness_dir / "USER.md"
        self.session_dir = config.session_memory_dir or self.ness_dir / "sessions"
        
    @property
    def disabled(self) -> bool:
        return self.cfg.disabled


    def _expand_includes(self, text:str) -> tuple[str, list[Path]]:
        """Inline standalone `@path` directives with the referenced file contents.

        Resolves relative to the project root, rejects escapes, skips missing files,
        guards against include cycles, and caps the total inlined size. Returns the
        expanded text and the ordered list of resolved files that were inlined (for
        cache invalidation)."""
        collected: list[Path] = []
        seen: set[Path] = set()
        budget = [MAX_NESS_INCLUDE_CHARS]
        root = self.ness_file.parent.resolve()

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
        """Load the project memory from the NESS.md file. Resolves @include directives."""
        if self.disabled or not self.ness_file.exists(): 
            return ""
        
        expanded, _ = self._expand_includes(self.ness_file.read_text(encoding="utf-8"))
        return expanded


    def append_project(self, text: str) -> str:
        """
        Append text to the NESS.md file. 
        Should be used for manual edits only. Or used by the cli with /memory {text}
        """
        if self.disabled: 
            return "disabled"
        return self._append_markdown(self.ness_file, text)


    def write_project(self, text: str, overwrite: bool = False) -> str:
        """
        Creates and writes text to the NESS.md file. 
        Overwrites if overwrite is True. (False by default)
        Or used by the cli with /init for general purpose Project Memory created by the agent.
        """
        if self.ness_file.exists() and not overwrite: 
            return f"Error: {self.ness_file} exists"
        
        self.ness_file.parent.mkdir(parents=True, exist_ok=True)
        self.ness_file.write_text(text.strip() + "\n", encoding="utf-8")
        return f"Wrote {self.ness_file}"


    # user memory (USER.md)
    def load_user(self) -> str:
        """Load the user memory from the USER.md file."""
        if self.disabled or not self.user_file.exists(): 
            return ""
        return self.user_file.read_text(encoding="utf-8")


    def append_user(self, text: str) -> str:
        """
        Append text to the USER.md file. 
        Should be used for manual edits only. Or used by the cli with /user {text}
        """
        if self.disabled: 
            return "disabled"
        return self._append_markdown(self.user_file, text)

    def write_user(self, text: str, overwrite: bool = False) -> str:
        """
        Creates and writes text to the USER.md file.
        Overwrites if overwrite is True. (False by default)
        """
        if self.user_file.exists() and not overwrite:
            return f"Error: {self.user_file} exists"

        self.user_file.parent.mkdir(parents=True, exist_ok=True)
        self.user_file.write_text(text.strip() + "\n", encoding="utf-8")
        return f"Wrote {self.user_file}"


    # session memory (sessions/mem_<thread_id>.md)
    def _session_path(self, thread_id: str) -> Path:
        """Get the path to the session memory file for the given thread_id."""
        return self.session_dir / f"mem_{thread_id}.md"

    def load_session(self, thread_id: str) -> str:
        """Load the session memory from the sessions/mem_<thread_id>.md file."""
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
        """Append bullet points to the sessions/mem_<thread_id>.md file."""
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
            if b not in merged: merged.append(b)
        text = "\n".join(f"- {b}" for b in merged)
        if text: 
            text += "\n"
        
        prev = p.read_text(encoding="utf-8") if p.exists() else ""
        if prev == text: 
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return True

    # setup
    def setup_structure(self) -> list[str]:
        """
        Setup the structure of the memory files.
        Creates the directories and files if they don't exist.
        The created files and directories are:
        - .ness/
        - .ness/sessions/
        - .ness/agents/
        - .ness/commands/
        - .ness/skills/
        - .ness/plans/
        - .ness/threads/
        - .ness/shells/
        - .ness/permissions.json
        - .ness/hooks.json
        - .ness/mcp.json

        Returns a list of created paths.
        """
        created = []
        if not self.ness_dir.exists():
            self.ness_dir.mkdir(parents=True, exist_ok=True)
            created.append(str(self.ness_dir))
        
        for name in NESS_SUBDIRS:
            p = self.ness_dir / name
            if not p.exists(): 
                p.mkdir(parents=True, exist_ok=True)
                created.append(str(p))
        
        for path, content in {
            self.ness_dir / "permissions.json": json.dumps(DEFAULT_RULES, indent=2) + "\n",
            self.ness_dir / "hooks.json": "{}\n",
            self.ness_dir / "mcp.json": json.dumps({"servers": {}}, indent=2) + "\n",
        }.items():
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
                created.append(str(path))
        
        return created

    def check_health(self) -> str | None:
        """
        Check the health of the project memory.
        Returns a warning if the project memory is at or above the threshold size.
        """
        chars = len(self.load_project())
        if chars > MAX_NESS_CHARS:
            return f"Warning: NESS.md is at {chars} chars (threshold {MAX_NESS_CHARS})."
        return None

    # helpers
    def _append_markdown(self, path: Path, text: str) -> str:
        cleaned = text.strip()
        if not cleaned: 
            return f"No changes for {path}"
        
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f: 
            f.write(cleaned + "\n")
        
        return f"Appended to {path}"