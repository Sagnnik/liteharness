from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import memory
import rollback
import session
from config import settings


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
        memory.NESS_DIR = self.ness_dir
        session.THREADS_DIR = self.ness_dir / "threads"
        session.THREADS_DB = session.THREADS_DIR / "threads.db"
        self._old_autosave = settings.auto_save_threads
        settings.auto_save_threads = True
        self.cwd = Path(self._tmpdir.name) / "repo"
        self.cwd.mkdir()
        _git_init(self.cwd)

    def tearDown(self) -> None:
        settings.auto_save_threads = self._old_autosave
        self._tmpdir.cleanup()

    def test_save_checkpoint_and_truncate(self) -> None:
        seq = session.append_event("session-rb", {"kind": "user", "content": "first turn"})
        session.save_checkpoint("session-rb", seq, git_hash="abc123", mem_snapshot="initial")
        checkpoint = session.get_checkpoint("session-rb", seq)
        self.assertEqual(checkpoint["git_hash"], "abc123")
        self.assertEqual(checkpoint["mem_snapshot"], "initial")

        session.append_event("session-rb", {"kind": "assistant", "content": "reply"})
        later = session.append_event("session-rb", {"kind": "user", "content": "second turn"})
        session.save_checkpoint("session-rb", later, git_hash="def456", mem_snapshot="later")
        session.truncate_after("session-rb", later)

        events = session.load_thread_events("session-rb")
        self.assertEqual([e["content"] for e in events if e["kind"] in {"user", "assistant"}], ["first turn", "reply"])
        self.assertIsNone(session.get_checkpoint("session-rb", later))
        self.assertIsNotNone(session.get_checkpoint("session-rb", seq))

    def test_restore_paths_surgical(self) -> None:
        (self.cwd / "src.txt").write_text("v1")
        h1 = rollback.create_file_checkpoint(self.cwd)
        (self.cwd / "src.txt").write_text("v2")

        rollback.restore_paths(h1, ["src.txt"], cwd=self.cwd)
        self.assertEqual((self.cwd / "src.txt").read_text(), "v1")

    def test_restore_mem_file_writes_snapshot(self) -> None:
        rollback.restore_mem_file(self.ness_dir, "session-x", "- a\n- b\n")
        path = self.ness_dir / "sessions" / "mem_session-x.md"
        self.assertTrue(path.exists())
        self.assertEqual(path.read_text(), "- a\n- b\n")


if __name__ == "__main__":
    unittest.main()
