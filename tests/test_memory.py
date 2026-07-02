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
        self.assertNotIn("## Session", loaded)

    def test_append_session_bullets_dedupes_and_appends(self) -> None:
        self.assertTrue(memory.append_session_bullets("thread-a", ["First note"]))
        self.assertTrue(memory.append_session_bullets("thread-a", ["First note", "Second note"]))
        loaded = memory.load_session_memory("thread-a")
        self.assertEqual(loaded.count("- First note"), 1)
        self.assertIn("- Second note", loaded)

    def test_memory_key_tracks_thread_file(self) -> None:
        self.assertEqual(memory.memory_key("thread-a"), (False, 0, 0))
        memory.append_session_bullets("thread-a", ["note"])
        key = memory.memory_key("thread-a")
        self.assertTrue(key[0])
        self.assertGreater(key[2], 0)

    def test_append_ness_memory_skips_empty_input(self) -> None:
        result = memory.append_ness_memory("   ")
        self.assertIn("No changes for", result)
        self.assertFalse((self.ness / "NESS.md").exists())

    def test_check_ness_health_warns_above_threshold(self) -> None:
        (self.ness / "NESS.md").write_text("x" * (memory.MAX_NESS_CHARS + 1), encoding="utf-8")
        warning = memory.check_ness_health()
        self.assertIsNotNone(warning)
        self.assertIn("Warning:", warning)


if __name__ == "__main__":
    unittest.main()
