"""Tests for path resolution."""

from __future__ import annotations

from pathlib import Path

from ness_cli.paths import (
    ensure_global_config,
    project_hash,
    resolve_paths,
    resolve_project_slug,
    sanitize_slug,
)


def test_sanitize_slug():
    assert sanitize_slug("My App!") == "my-app"
    assert sanitize_slug("___") == "project"


def test_resolve_paths_uses_env_overrides(tmp_path: Path, monkeypatch):
    project = tmp_path / "myproj"
    project.mkdir()
    cfg = tmp_path / "config"
    cache = tmp_path / "cache"
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(cfg))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(cache))
    monkeypatch.setenv("NESS_DIR", ".ness")

    paths = resolve_paths(project_root=project)
    assert paths.project_root == project.resolve()
    assert paths.ness_dir == (project / ".ness").resolve()
    assert paths.config_dir == cfg.resolve()
    assert paths.user_file == cfg.resolve() / "USER.md"
    assert paths.configs_file == cfg.resolve() / "configs.json"
    assert paths.secrets_file == cfg.resolve() / "secrets.json"
    assert paths.instructions_dir == cfg.resolve() / "instructions"
    assert paths.plans_dir == cfg.resolve() / "plans" / "myproj"
    assert paths.sessions_dir == paths.ness_dir / "runtime" / "sessions"
    assert paths.shells_dir == paths.ness_dir / "runtime" / "shells"
    assert paths.threads_dir == paths.ness_dir / "threads"
    assert paths.cli_history == cache.resolve() / paths.project_hash / "cli_history"
    assert paths.project_hash == project_hash(project)


def test_slug_collision_appends_hash(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))

    a = tmp_path / "apps" / "demo"
    b = tmp_path / "other" / "demo"
    a.mkdir(parents=True)
    b.mkdir(parents=True)

    plans_root = tmp_path / "cfg" / "plans"
    plans_root.mkdir(parents=True)
    taken = plans_root / "demo"
    taken.mkdir()
    (taken / ".project").write_text(str(a.resolve()) + "\n", encoding="utf-8")

    slug_a = resolve_project_slug(a, plans_root)
    slug_b = resolve_project_slug(b, plans_root)
    assert slug_a == "demo"
    assert slug_b.startswith("demo-")
    assert slug_b != slug_a


def test_ensure_global_config_creates_user_and_marker(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "repo"
    project.mkdir()
    paths = resolve_paths(project_root=project)
    created = ensure_global_config(paths)
    assert paths.user_file.is_file()
    assert (paths.plans_dir / ".project").is_file()
    assert any("USER.md" in c for c in created)
    # secrets.json is created eagerly with restrictive perms; configs.json is lazy.
    assert paths.secrets_file.is_file()
    import json
    import stat

    assert json.loads(paths.secrets_file.read_text()) == {}
    assert stat.S_IMODE(paths.secrets_file.stat().st_mode) == 0o600
    assert not paths.configs_file.exists()
    assert paths.instructions_dir.is_dir()
    from ness_cli.instructions import INSTRUCTION_FILES

    for name in INSTRUCTION_FILES:
        assert (paths.instructions_dir / name).is_file()
        assert (paths.instructions_dir / name).read_text(encoding="utf-8").strip()


def test_ensure_global_config_does_not_overwrite_instructions(
    tmp_path: Path, monkeypatch
):
    monkeypatch.setenv("NESS_AI_CONFIG_DIR", str(tmp_path / "cfg"))
    monkeypatch.setenv("NESS_AI_CACHE_DIR", str(tmp_path / "cache"))
    project = tmp_path / "repo"
    project.mkdir()
    paths = resolve_paths(project_root=project)
    ensure_global_config(paths)
    custom = "CUSTOM L0\n"
    target = paths.instructions_dir / "l0_harness.md"
    target.write_text(custom, encoding="utf-8")
    ensure_global_config(paths)
    assert target.read_text(encoding="utf-8") == custom
