from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from liteharness.memory import MemoryStore
from liteharness.options import MemoryConfig


class MemoryModuleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.ness = Path(self.tmp.name) / ".ness"
        self.ness.mkdir()
        self.store = MemoryStore(
            MemoryConfig(),
            ness_dir=self.ness,
            project_root=Path(self.tmp.name),
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_session_memory_returns_bullets_only(self) -> None:
        self.store.append_session_bullets("thread-a", ["Built auth middleware"])
        loaded = self.store.load_session("thread-a")
        self.assertIn("- Built auth middleware", loaded)
        self.assertNotIn("thread-a", loaded)

    def test_append_session_bullets_dedupes_and_appends(self) -> None:
        self.assertTrue(self.store.append_session_bullets("thread-a", ["First note"]))
        self.assertTrue(self.store.append_session_bullets("thread-a", ["First note", "Second note"]))
        loaded = self.store.load_session("thread-a")
        self.assertEqual(loaded.count("- First note"), 1)
        self.assertIn("- Second note", loaded)


if __name__ == "__main__":
    unittest.main()
