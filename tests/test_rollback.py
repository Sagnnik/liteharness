from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ness_ai.persistence import ThreadStore
import ness_cli.rollback as rollback


def _git_init(path: Path) -> None:
    import subprocess

    subprocess.run(["git", "init", "-q"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.email", "t@t"], cwd=path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=path, check=True)
    subprocess.run(["git", "commit", "--allow-empty", "-q", "-m", "init"], cwd=path, check=True)


class RollbackSmokeTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.ness_dir = Path(self._tmpdir.name) / "ness"
        self.ness_dir.mkdir()
        self.store = ThreadStore(threads_dir=self.ness_dir / "threads", auto_save=True)
        self.cwd = Path(self._tmpdir.name) / "repo"
        self.cwd.mkdir()
        _git_init(self.cwd)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_save_checkpoint_and_truncate(self) -> None:
        seq = self.store.append_event("session-rb", {"kind": "user", "content": "first turn"})
        self.store.save_checkpoint("session-rb", seq, git_hash="abc123", mem_snapshot="initial")
        checkpoint = self.store.get_checkpoint("session-rb", seq)
        self.assertEqual(checkpoint["git_hash"], "abc123")
        self.assertEqual(checkpoint["mem_snapshot"], "initial")

        self.store.append_event("session-rb", {"kind": "assistant", "content": "reply"})
        later = self.store.append_event("session-rb", {"kind": "user", "content": "second turn"})
        self.store.save_checkpoint("session-rb", later, git_hash="def456", mem_snapshot="later")
        self.store.truncate_after("session-rb", later)

        events = self.store.load_thread_events("session-rb")
        self.assertEqual(
            [e["content"] for e in events if e["kind"] in {"user", "assistant"}],
            ["first turn", "reply"],
        )
        self.assertIsNone(self.store.get_checkpoint("session-rb", later))
        self.assertIsNotNone(self.store.get_checkpoint("session-rb", seq))

    def test_restore_paths_surgical(self) -> None:
        (self.cwd / "src.txt").write_text("v1")
        h1 = rollback.create_file_checkpoint(self.cwd)
        (self.cwd / "src.txt").write_text("v2")

        rollback.restore_paths(h1, ["src.txt"], cwd=self.cwd)
        self.assertEqual((self.cwd / "src.txt").read_text(), "v1")

    def test_restore_mem_file_writes_snapshot(self) -> None:
        rollback.restore_mem_file(self.ness_dir, "session-x", "- a\n- b\n")
        path = self.ness_dir / "runtime" / "sessions" / "mem_session-x.md"
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "- a\n- b\n")


if __name__ == "__main__":
    unittest.main()
