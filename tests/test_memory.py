from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import memory


class MemoryModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ness = Path(self.tmp.name) / ".ness"
        self.ness.mkdir()
        self._patch_ness_dir()

    def tearDown(self) -> None:
        for patcher in getattr(self, "patchers", []):
            patcher.stop()
        self.tmp.cleanup()

    def _patch_ness_dir(self) -> None:
        import unittest.mock as mock

        sessions_dir = self.ness / "sessions"
        self.patchers = [
            mock.patch.object(memory, "NESS_DIR", self.ness),
            mock.patch.object(memory, "NESS_FILE", self.ness / "NESS.md"),
            mock.patch.object(memory, "SESSIONS_DIR", sessions_dir),
        ]
        for patcher in self.patchers:
            patcher.start()

    def test_load_session_memory_returns_bullets_only(self) -> None:
        memory.append_session_bullets("thread-a", ["Built auth middleware"])
        loaded = memory.load_session_memory("thread-a")
        self.assertIn("- Built auth middleware", loaded)
        self.assertNotIn("thread-a", loaded)

    def test_append_session_bullets_dedupes_and_appends(self) -> None:
        self.assertTrue(memory.append_session_bullets("thread-a", ["First note"]))
        self.assertTrue(memory.append_session_bullets("thread-a", ["First note", "Second note"]))
        loaded = memory.load_session_memory("thread-a")
        self.assertEqual(loaded.count("- First note"), 1)
        self.assertIn("- Second note", loaded)


if __name__ == "__main__":
    unittest.main()
