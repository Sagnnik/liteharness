import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")

from worktree import WorktreeError, ensure_worktree, repo_root, slugify, worktree_path


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )


class WorktreeHelperTests(unittest.TestCase):
    def test_slugify_normalizes_names(self) -> None:
        self.assertEqual(slugify("Feature Auth"), "feature-auth")
        self.assertEqual(slugify("bug_fix"), "bug-fix")

    def test_slugify_rejects_empty(self) -> None:
        with self.assertRaises(WorktreeError):
            slugify("!!!")

    def test_repo_root_outside_git(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(repo_root(Path(tmp)))


class EnsureWorktreeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name)
        _git(self.root, "init")
        _git(self.root, "config", "user.email", "test@example.com")
        _git(self.root, "config", "user.name", "Test")
        (self.root / "README.md").write_text("# test\n", encoding="utf-8")
        _git(self.root, "add", "README.md")
        _git(self.root, "commit", "-m", "init")

    def tearDown(self) -> None:
        for path in sorted(self.root.glob(".ness/worktrees/*")):
            if path.is_dir():
                subprocess.run(
                    ["git", "worktree", "remove", "--force", str(path)],
                    cwd=self.root,
                    capture_output=True,
                )
                branch = f"worktree-{path.name}"
                subprocess.run(
                    ["git", "branch", "-D", branch],
                    cwd=self.root,
                    capture_output=True,
                )
        self._tmpdir.cleanup()

    def test_ensure_worktree_creates_branch_and_path(self) -> None:
        original = Path.cwd()
        try:
            os.chdir(self.root)
            path = ensure_worktree("auth")
            self.assertTrue(path.is_dir())
            self.assertEqual(path, worktree_path("auth"))

            wt_branch = _git(path, "branch", "--show-current")
            self.assertEqual(wt_branch.stdout.strip(), "worktree-auth")
        finally:
            os.chdir(original)

    def test_ensure_worktree_is_idempotent(self) -> None:
        original = Path.cwd()
        try:
            os.chdir(self.root)
            first = ensure_worktree("auth")
            second = ensure_worktree("auth")
            self.assertEqual(first, second)
        finally:
            os.chdir(original)

    def test_ensure_worktree_copies_env(self) -> None:
        original = Path.cwd()
        try:
            (self.root / ".env").write_text("OPENAI_API_KEY=copied\n", encoding="utf-8")
            os.chdir(self.root)
            path = ensure_worktree("env-copy")
            copied = path / ".env"
            self.assertTrue(copied.is_file())
            self.assertIn("OPENAI_API_KEY=copied", copied.read_text(encoding="utf-8"))
        finally:
            os.chdir(original)

    def test_ensure_worktree_requires_git_repo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            original = Path.cwd()
            try:
                os.chdir(tmp)
                with self.assertRaises(WorktreeError):
                    ensure_worktree("nope")
            finally:
                os.chdir(original)


if __name__ == "__main__":
    unittest.main()
