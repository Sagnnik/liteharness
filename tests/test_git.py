import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")

import tools.git  # noqa: F401  (ensure submodule is imported)
from tools.git import (
    auto_git_snapshot,
    git_worktree_summary,
)
from tools.git import git as git_tool

# `tools/__init__.py` does `from tools.git import git`, which rebinds the
# `tools.git` package attribute to the tool function. Fetch the real module
# object from sys.modules so mock.patch.object targets the module, not the tool.
git = sys.modules["tools.git"]


class AutoGitSnapshotTests(unittest.TestCase):
    def test_returns_true_on_successful_commit(self) -> None:
        with mock.patch.object(git, "_git", side_effect=["(ok)", "committed"]):
            self.assertTrue(auto_git_snapshot())

    def test_returns_true_when_nothing_to_commit(self) -> None:
        with mock.patch.object(
            git,
            "_git",
            side_effect=["(ok)", "Error: git commit failed\nnothing to commit"],
        ):
            self.assertTrue(auto_git_snapshot())

    def test_returns_false_when_add_fails(self) -> None:
        with mock.patch.object(git, "_git", return_value="Error: git add failed"):
            self.assertFalse(auto_git_snapshot())

    def test_uses_no_verify_on_commit(self) -> None:
        with mock.patch.object(git, "_git", side_effect=["(ok)", "(ok)"]) as mock_git:
            auto_git_snapshot("agent: test")
        mock_git.assert_any_call(["commit", "-m", "agent: test", "--no-verify"], timeout=60)


class GitToolArgTests(unittest.TestCase):
    def test_git_diff_stat_flag(self) -> None:
        with mock.patch.object(git, "_git", return_value="stat output") as mock_git:
            git_tool.invoke({"action": "diff", "stat": True})
        mock_git.assert_called_once_with(["diff", "--stat"])

    def test_git_log_grep_flag(self) -> None:
        with mock.patch.object(git, "_git", return_value="log output") as mock_git:
            git_tool.invoke({"action": "log", "grep": "fix bug"})
        mock_git.assert_called_once_with(["log", "-20", "--oneline", "--grep", "fix bug"])

    def test_git_log_clamps_count(self) -> None:
        with mock.patch.object(git, "_git", return_value="log output") as mock_git:
            git_tool.invoke({"action": "log", "n": 500})
        mock_git.assert_called_once_with(["log", "-100", "--oneline"])

        with mock.patch.object(git, "_git", return_value="log output") as mock_git:
            git_tool.invoke({"action": "log", "n": 0})
        mock_git.assert_called_once_with(["log", "-1", "--oneline"])

        with mock.patch.object(git, "_git", return_value="log output") as mock_git:
            git_tool.invoke({"action": "log", "n": "not-a-number"})
        mock_git.assert_called_once_with(["log", "-20", "--oneline"])

    def test_git_show_rejects_option_like_revision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(git, "_git") as mock_git:
            output = Path(tmp) / "show.txt"
            result = git_tool.invoke({"action": "show", "rev": f"--output={output}"})
        self.assertIn("must not start", result)
        self.assertFalse(output.exists())
        mock_git.assert_not_called()

    def test_git_show_uses_metadata_only_flags(self) -> None:
        with mock.patch.object(git, "_git", return_value="commit metadata") as mock_git:
            git_tool.invoke({"action": "show", "rev": "HEAD"})
        mock_git.assert_called_once_with(["show", "--no-ext-diff", "--no-patch", "HEAD"])

    def test_git_checkout_rejects_unsafe_branch_names(self) -> None:
        for branch in ("", "--orphan", "feature bad", "bad\nname"):
            with self.subTest(branch=branch), mock.patch.object(git, "_git") as mock_git:
                result = git_tool.invoke({"action": "checkout", "branch": branch})
                self.assertTrue(result.startswith("Error:"))
                mock_git.assert_not_called()

    def test_git_branch_rejects_unsafe_names_but_lists_without_name(self) -> None:
        with mock.patch.object(git, "_git", return_value="* main") as mock_git:
            self.assertEqual(git_tool.invoke({"action": "branch"}), "* main")
        mock_git.assert_called_once_with(["branch"])

        for name in ("-bad", "feature bad", "bad\tname"):
            with self.subTest(name=name), mock.patch.object(git, "_git") as mock_git:
                result = git_tool.invoke({"action": "branch", "name": name})
                self.assertTrue(result.startswith("Error:"))
                mock_git.assert_not_called()

    def test_git_stash_push_requires_message(self) -> None:
        result = git_tool.invoke({"action": "stash", "stash_action": "push"})
        self.assertIn("requires a message", result)

    def test_git_stash_push_with_message(self) -> None:
        with mock.patch.object(git, "_git", return_value="(ok)") as mock_git:
            git_tool.invoke({"action": "stash", "stash_action": "push", "message": "wip"})
        mock_git.assert_called_once_with(["stash", "push", "-m", "wip"])

    def test_git_commit_requires_message(self) -> None:
        result = git_tool.invoke({"action": "commit"})
        self.assertIn("requires a message", result)


class GitWorktreeSummaryTests(unittest.TestCase):
    def test_clean_tree(self) -> None:
        with mock.patch.object(git, "_git", side_effect=["main", "(ok)"]):
            self.assertEqual(git_worktree_summary(), "branch: main; working tree clean")

    def test_dirty_tree_truncates_paths(self) -> None:
        porcelain = "\n".join(f" M file{i}.py" for i in range(8))
        with mock.patch.object(git, "_git", side_effect=["dev", porcelain]):
            summary = git_worktree_summary()
        self.assertIn("branch: dev", summary)
        self.assertIn("8 changed file(s)", summary)
        self.assertIn("file0.py", summary)
        self.assertIn("(+3 more)", summary)

    def test_returns_empty_when_not_in_repo(self) -> None:
        with mock.patch.object(
            git,
            "_git",
            return_value="Error: git branch --show-current failed with exit 128",
        ):
            self.assertEqual(git_worktree_summary(), "")

    def test_detached_head(self) -> None:
        with mock.patch.object(git, "_git", side_effect=["(ok)", "(ok)"]):
            self.assertEqual(git_worktree_summary(), "branch: (detached); working tree clean")


if __name__ == "__main__":
    unittest.main()
