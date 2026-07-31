"""Central path resolution for Ness (global config, project .ness, cache).

Resolves once at CLI startup and feeds concrete paths into the SDK /
adapter. SDK code keeps accepting overrides; it does not call platformdirs.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir

_APP_NAME = "ness-agent"
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_PROJECT_MARKER = ".project"


@dataclass(frozen=True)
class NessPaths:
    """Resolved filesystem locations for one project checkout."""

    project_root: Path
    ness_dir: Path
    config_dir: Path
    cache_dir: Path
    user_file: Path
    configs_file: Path
    secrets_file: Path
    instructions_dir: Path
    plans_dir: Path
    sessions_dir: Path
    shells_dir: Path
    threads_dir: Path
    cli_history: Path
    project_slug: str
    project_hash: str


def project_hash(project_root: Path) -> str:
    """Stable short hash of the resolved project root (for cache dirs)."""
    digest = hashlib.sha256(str(project_root.resolve()).encode("utf-8")).hexdigest()
    return digest[:12]


def sanitize_slug(name: str) -> str:
    """Return a filesystem-safe lowercase slug from a directory name."""
    slug = _SLUG_RE.sub("-", name.lower()).strip("-")
    return slug or "project"


def resolve_project_slug(
    project_root: Path,
    plans_root: Path,
    *,
    hash_hex: str | None = None,
) -> str:
    """Pick ``plans/<slug>/``; append ``-<hash6>`` on cross-project collision."""
    root = project_root.resolve()
    h = hash_hex or project_hash(root)
    base = sanitize_slug(root.name)
    candidate = plans_root / base
    marker = candidate / _PROJECT_MARKER
    root_str = str(root)

    if not candidate.exists():
        return base
    if marker.is_file():
        try:
            if marker.read_text(encoding="utf-8").strip() == root_str:
                return base
        except OSError:
            pass
    # Directory exists for a different project (or unmarked legacy content).
    return f"{base}-{h[:6]}"


def config_dir_from_env() -> Path:
    override = os.environ.get("NESS_AGENT_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_config_dir(_APP_NAME))


def cache_dir_from_env() -> Path:
    override = os.environ.get("NESS_AGENT_CACHE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path(user_cache_dir(_APP_NAME))


def resolve_paths(
    *,
    project_root: Path | None = None,
    ness_dir: Path | str | None = None,
) -> NessPaths:
    """Resolve all Ness paths for the current (or given) project root."""
    root = (project_root or Path.cwd()).resolve()
    if ness_dir is None:
        ness_dir = Path(os.environ.get("NESS_DIR", ".ness"))
    ness = Path(ness_dir)
    if not ness.is_absolute():
        ness = (root / ness).resolve()
    else:
        ness = ness.resolve()

    cfg = config_dir_from_env()
    cache_root = cache_dir_from_env()
    h = project_hash(root)
    plans_root = cfg / "plans"
    slug = resolve_project_slug(root, plans_root, hash_hex=h)
    project_cache = cache_root / h

    return NessPaths(
        project_root=root,
        ness_dir=ness,
        config_dir=cfg,
        cache_dir=project_cache,
        user_file=cfg / "USER.md",
        configs_file=cfg / "configs.json",
        secrets_file=cfg / "secrets.json",
        instructions_dir=cfg / "instructions",
        plans_dir=plans_root / slug,
        sessions_dir=ness / "runtime" / "sessions",
        shells_dir=ness / "runtime" / "shells",
        threads_dir=ness / "threads",
        cli_history=project_cache / "cli_history",
        project_slug=slug,
        project_hash=h,
    )


def ensure_global_config(paths: NessPaths) -> list[str]:
    """Create global config dir, plans slug dir + marker, empty USER.md,
    ``secrets.json``, and instruction templates if missing.

    ``configs.json`` is created lazily on first write (see
    :mod:`ness_cli.config_store`). Returns a list of paths that were
    created.
    """
    created: list[str] = []
    if not paths.config_dir.exists():
        paths.config_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.config_dir))

    if not paths.user_file.exists():
        paths.user_file.parent.mkdir(parents=True, exist_ok=True)
        paths.user_file.write_text("", encoding="utf-8")
        created.append(str(paths.user_file))

    from ness_cli.config_store import ensure_secrets_file

    secrets_created = ensure_secrets_file(paths.config_dir)
    if secrets_created is not None:
        created.append(str(secrets_created))

    if not paths.plans_dir.exists():
        paths.plans_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.plans_dir))

    marker = paths.plans_dir / _PROJECT_MARKER
    root_str = str(paths.project_root.resolve())
    if not marker.exists():
        marker.write_text(root_str + "\n", encoding="utf-8")
        created.append(str(marker))
    else:
        try:
            if marker.read_text(encoding="utf-8").strip() != root_str:
                marker.write_text(root_str + "\n", encoding="utf-8")
        except OSError:
            marker.write_text(root_str + "\n", encoding="utf-8")

    created.extend(_ensure_instruction_files(paths))

    return created


def _ensure_instruction_files(paths: NessPaths) -> list[str]:
    """Seed packaged ``instructions/*.md`` into the global config dir."""
    from ness_cli.instructions import default_instruction_files

    created: list[str] = []
    if not paths.instructions_dir.exists():
        paths.instructions_dir.mkdir(parents=True, exist_ok=True)
        created.append(str(paths.instructions_dir))

    for filename, content in default_instruction_files().items():
        path = paths.instructions_dir / filename
        if path.exists():
            continue
        path.write_text(content, encoding="utf-8")
        created.append(str(path))
    return created


def ensure_project_runtime(paths: NessPaths) -> list[str]:
    """Create runtime session/shell dirs under the project ``.ness/``."""
    created: list[str] = []
    for d in (paths.sessions_dir, paths.shells_dir, paths.threads_dir):
        if not d.exists():
            d.mkdir(parents=True, exist_ok=True)
            created.append(str(d))
    return created
