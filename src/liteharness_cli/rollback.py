"""Shadow-git and session-memory snapshot helpers for state rollback.

Stdlib-only: mirrors ``worktree.py``'s constraint of not importing config or
permissions (the workspace cwd is set by the caller). All git operations run
against the current working directory, which is the main checkout in plain
mode or the worktree path under ``.ness/worktrees/<slug>`` in worktree mode.

Design notes:

- ``create_file_checkpoint`` builds a *rootless tree object* capturing the full
  working-tree state (tracked modifications + untracked files + deletions)
  WITHOUT touching the user's index, working tree, or branch. It does this by
  writing to a throwaway index file (``GIT_INDEX_FILE``) via ``read-tree HEAD``
  then ``add -A`` then ``write-tree``. The returned hash is the tree SHA, which
  ``git checkout <hash> -- <paths>`` will happily restore from. Tree objects
  are unreferenced and will be GC'd eventually by git, so this is non-invasive
  in both directions.

  ``git stash create --include-untracked`` was rejected because it returns
  nothing when the working tree differs from HEAD only by *untracked* files
  (no tracked modifications), which is common mid-session.

  Returns the literal string ``"HEAD"`` when the workspace is clean vs HEAD
  (no snapshot is needed; restoring is equivalent to checking out HEAD), and
  ``None`` when not in a git repo (rollback degrades gracefully: only the
  conversation + session memory are restored then).

- ``restore_paths`` is surgical: each path in the list is restored from the
  tree if the tree contains it, or deleted from the working tree if it doesn't
  (the agent created it after the snapshot). Empty list or the ``"*"`` sentinel
  triggers a full-tree restore via ``git checkout <hash> -- .``.

- ``restore_mem_file`` writes the per-thread session memory file
  (``sessions/mem_<thread_id>.md``) from the snapshot stored at the checkpoint.
  ``append_session_bullets`` is append-only, so restoring the snapshot undoes
  any bullets the abandoned-turn reflection wrote.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path


def _run_git(args: list[str], *, cwd: Path, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    full_env = dict(os.environ)
    if env:
        full_env.update(env)
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
        env=full_env,
    )


def is_git_workspace(cwd: Path | None = None) -> bool:
    result = _run_git(["rev-parse", "--is-inside-work-tree"], cwd=cwd or Path.cwd())
    return result.returncode == 0 and result.stdout.strip() == "true"


def create_file_checkpoint(cwd: Path | None = None) -> str | None:
    """Snapshot the workspace as a rootless tree; return its SHA, "HEAD", or None.

    Returns:
      - a tree SHA when the workspace has uncommitted working-tree state
      - the literal ``"HEAD"`` when the workspace is clean vs HEAD
      - ``None`` when not in a git repo (rollback degrades to conversation-only)
    """
    work = cwd or Path.cwd()
    if not is_git_workspace(work):
        return None

    # Fast path: if there is nothing to snapshot (no tracked/untracked delta),
    # restoring is the same as checking out HEAD.
    status = _run_git(["status", "--porcelain"], cwd=work)
    if status.returncode == 0 and not status.stdout.strip():
        return "HEAD"

    # Build a throwaway index from HEAD, then add the entire working tree
    # (including untracked files), then materialize a tree object. The user's
    # real index file and branch are never touched.
    tmp_index = tempfile.NamedTemporaryFile(prefix="lh-shadow-", suffix=".idx", delete=False)
    tmp_index.close()
    try:
        env = {"GIT_INDEX_FILE": tmp_index.name}
        read = _run_git(["read-tree", "HEAD"], cwd=work, env=env)
        if read.returncode != 0:
            return None
        add = _run_git(["add", "-A"], cwd=work, env=env)
        if add.returncode != 0:
            return None
        write = _run_git(["write-tree"], cwd=work, env=env)
        if write.returncode != 0:
            return None
        tree = write.stdout.strip()
        return tree or None
    finally:
        try:
            os.unlink(tmp_index.name)
        except OSError:
            pass


def _path_in_tree(tree_or_ref: str, path: str, cwd: Path) -> bool:
    # `git ls-tree -- <path>` prints the entry if it's in the tree, empty
    # otherwise; exit code is 0 in both cases.
    result = _run_git(["ls-tree", tree_or_ref, "--", path], cwd=cwd)
    return bool(result.stdout.strip())


def restore_paths(git_hash: str, paths: list[str], cwd: Path | None = None) -> str:
    """Restore the given paths from the snapshot. Empty list / "*" = full tree.

    Surgical restore (a path list):
      - if the path is in the snapshot tree -> ``git checkout <hash> -- <path>``
        (restores the file to its state at snapshot time, creating/overwriting).
      - else the path was created AFTER the snapshot (by the agent); delete the
        current file if it exists, so the working tree matches the snapshot.

    Full-tree restore (empty list or ``"*"``):
      - ``git checkout <hash> -- .`` overwrites tracked files with the snapshot
        state. Untracked files created after the snapshot are NOT deleted (git
        won't touch them). Matches the reviewer's original semantics.
    """
    work = cwd or Path.cwd()
    if git_hash == "HEAD" and not paths:
        result = _run_git(["checkout", git_hash, "--", "."], cwd=work)
        return (result.stdout + result.stderr).strip() or "(ok)"
    if not paths or "*" in paths:
        result = _run_git(["checkout", git_hash, "--", "."], cwd=work)
        return (result.stdout + result.stderr).strip() or "(ok)"

    notes: list[str] = []
    for path in paths:
        if _path_in_tree(git_hash, path, work):
            result = _run_git(["checkout", git_hash, "--", path], cwd=work)
            if result.returncode != 0:
                msg = (result.stderr or result.stdout).strip()
                if msg:
                    notes.append(msg)
        else:
            # Path was created after the snapshot (likely by the agent) and is
            # not in the snapshot tree nor in HEAD's tree. Delete it so the
            # working tree matches the snapshot state.
            abs_path = (work / path).resolve()
            try:
                abs_path.relative_to(work.resolve())
            except ValueError:
                continue  # never delete outside the workspace
            try:
                if abs_path.exists() and abs_path.is_file():
                    abs_path.unlink()
            except OSError as exc:
                notes.append(f"could not delete {path}: {exc}")
    return "\n".join(notes)


def restore_mem_file(ness_dir: Path, thread_id: str, snapshot: str) -> None:
    """Overwrite the per-thread session memory file from a checkpoint snapshot.

    An empty snapshot deletes the file (matches ``append_session_bullets``'
    empty-state semantics). No-op when ``snapshot`` is None.
    """
    if snapshot is None:
        return
    path = ness_dir / "sessions" / f"mem_{thread_id}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    if snapshot:
        path.write_text(snapshot, encoding="utf-8")
    elif path.exists():
        path.unlink()


def read_mem_file(ness_dir: Path, thread_id: str) -> str:
    """Read the current per-thread session memory file for snapshotting."""
    path = ness_dir / "sessions" / f"mem_{thread_id}.md"
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return ""