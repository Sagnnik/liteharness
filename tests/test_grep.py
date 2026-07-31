from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ness_ai.tools.search import grep

from tests.sdk_fixtures import SessionContextTestMixin


class GrepGlobFilterTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name))
        (self.root / "a.py").write_text("needle_py\n", encoding="utf-8")
        (self.root / "b.txt").write_text("needle_txt\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_glob_filters_filenames(self) -> None:
        result = grep.invoke({"pattern": "needle", "glob": "*.py"})
        self.assertIn("needle_py", result)
        self.assertNotIn("needle_txt", result)

    def test_schema_exposes_glob_not_include(self) -> None:
        schema = grep.args_schema.model_json_schema()
        props = schema.get("properties", {})
        self.assertIn("glob", props)
        self.assertNotIn("include", props)


if __name__ == "__main__":
    unittest.main()
