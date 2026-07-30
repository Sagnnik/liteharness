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

    def test_write_project_creates_when_missing(self) -> None:
        result = self.store.write_project("hello conventions")
        self.assertTrue(result.startswith("Wrote"))
        self.assertEqual(self.store.ness_file.read_text(encoding="utf-8"), "hello conventions\n")

    def test_write_project_allows_empty_without_overwrite(self) -> None:
        self.store.ness_file.write_text("", encoding="utf-8")
        result = self.store.write_project("drafted")
        self.assertTrue(result.startswith("Wrote"))
        self.assertEqual(self.store.ness_file.read_text(encoding="utf-8"), "drafted\n")

    def test_write_project_allows_whitespace_only_without_overwrite(self) -> None:
        self.store.ness_file.write_text("  \n\t\n", encoding="utf-8")
        result = self.store.write_project("drafted")
        self.assertTrue(result.startswith("Wrote"))
        self.assertEqual(self.store.ness_file.read_text(encoding="utf-8"), "drafted\n")

    def test_write_project_blocks_nonempty_without_overwrite(self) -> None:
        self.store.ness_file.write_text("existing\n", encoding="utf-8")
        result = self.store.write_project("new")
        self.assertTrue(result.startswith("Error:"))
        self.assertEqual(self.store.ness_file.read_text(encoding="utf-8"), "existing\n")

    def test_write_project_force_overwrites_nonempty(self) -> None:
        self.store.ness_file.write_text("existing\n", encoding="utf-8")
        result = self.store.write_project("replaced", overwrite=True)
        self.assertTrue(result.startswith("Wrote"))
        self.assertEqual(self.store.ness_file.read_text(encoding="utf-8"), "replaced\n")


if __name__ == "__main__":
    unittest.main()
