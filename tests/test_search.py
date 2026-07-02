import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")

import permissions
from tools.search import grep


class GrepTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._root_patch = mock.patch.object(permissions, "PROJECT_ROOT", self.root)
        self._root_patch.start()

    def tearDown(self) -> None:
        self._root_patch.stop()
        self._tmp.cleanup()

    def test_ripgrep_uses_separator_before_option_like_pattern(self) -> None:
        completed = mock.Mock(returncode=0, stdout="file.txt:1:-needle\n", stderr="")
        with (
            mock.patch("tools.search.shutil.which", return_value="/usr/bin/rg"),
            mock.patch("tools.search.subprocess.run", return_value=completed) as run,
        ):
            result = grep.invoke({"pattern": "-needle", "path": "."})
        self.assertIn("-needle", result)
        cmd = run.call_args.args[0]
        separator = cmd.index("--")
        self.assertEqual(cmd[separator + 1], "-needle")
        self.assertEqual(cmd[separator + 2], str(self.root))

    def test_python_fallback_honors_nested_globs(self) -> None:
        target = self.root / "src" / "pkg" / "app.py"
        target.parent.mkdir(parents=True)
        target.write_text("needle\n", encoding="utf-8")
        (self.root / "notes.txt").write_text("needle\n", encoding="utf-8")

        with mock.patch("tools.search.shutil.which", return_value=None):
            result = grep.invoke({"pattern": "needle", "path": ".", "glob": "**/*.py"})

        self.assertIn("src/pkg/app.py:1: needle", result)
        self.assertNotIn("notes.txt", result)

    def test_python_fallback_caps_matches(self) -> None:
        target = self.root / "matches.txt"
        target.write_text("\n".join(f"needle {idx}" for idx in range(250)), encoding="utf-8")

        with mock.patch("tools.search.shutil.which", return_value=None):
            result = grep.invoke({"pattern": "needle", "path": "."})

        match_lines = [line for line in result.splitlines() if line.startswith("matches.txt:")]
        self.assertEqual(len(match_lines), 200)
        self.assertIn("truncated after 200 matches", result)


if __name__ == "__main__":
    unittest.main()
