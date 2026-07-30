"""Packaged CLI instruction defaults (seeded into global config as ``*.md``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_INSTRUCTIONS_DIR = Path(__file__).resolve().parent

# Filenames relative to the instructions package / global instructions dir.
INSTRUCTION_FILES: tuple[str, ...] = (
    "l0_harness.md",
    "persona.md",
    "plan_mode.md",
    "act_mode.md",
    "compaction.md",
    "reflection.md",
    "subagent.md",
    "thread_summary.md",
    "init_memory.md",
    "goal_judge.md",
    "goal_repair.md",
    "goal_generic_repair.md",
)


def default_instruction_files() -> dict[str, str]:
    """Return ``{filename: content}`` for built-in instruction templates.

    Filenames are relative to the global ``instructions/`` dir
    (e.g. ``l0_harness.md``).
    """
    files: dict[str, str] = {}
    for name in INSTRUCTION_FILES:
        path = _INSTRUCTIONS_DIR / name
        if path.is_file():
            files[name] = path.read_text(encoding="utf-8")
    return files


@lru_cache(maxsize=None)
def packaged_instruction(name: str) -> str:
    """Load one packaged instruction by filename (e.g. ``l0_harness.md``)."""
    path = _INSTRUCTIONS_DIR / name
    return path.read_text(encoding="utf-8").strip()


@lru_cache(maxsize=None)
def _read_instruction_file(path: str) -> str:
    """Cached UTF-8 read of an absolute instruction file path."""
    return Path(path).read_text(encoding="utf-8").strip()


def load_instruction(name: str, *, instructions_dir: Path | None = None) -> str:
    """Load an instruction from *instructions_dir*, falling back to packaged text.

    When *instructions_dir* is omitted, uses the global config
    ``instructions/`` directory (see :func:`liteharness_cli.paths.config_dir_from_env`).

    Successful file reads are cached for the process lifetime (edits require a
    restart, or :func:`clear_instruction_cache`, to take effect).
    """
    directory = instructions_dir
    if directory is None:
        from liteharness_cli.paths import config_dir_from_env

        directory = config_dir_from_env() / "instructions"
    path = directory / name
    try:
        if path.is_file():
            text = _read_instruction_file(str(path.resolve()))
            if text:
                return text
    except OSError:
        pass
    return packaged_instruction(name)


def clear_instruction_cache() -> None:
    """Drop cached instruction file reads (for tests / forced reload)."""
    _read_instruction_file.cache_clear()
    packaged_instruction.cache_clear()


__all__ = [
    "INSTRUCTION_FILES",
    "default_instruction_files",
    "packaged_instruction",
    "load_instruction",
    "clear_instruction_cache",
]
