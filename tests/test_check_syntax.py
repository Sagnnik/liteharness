import json
import importlib
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("OPENAI_API_KEY", "test")

import permissions
from tools.check_syntax import check_syntax

syntax = importlib.import_module("tools.check_syntax")


class CheckSyntaxTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.outer = Path(self._tmp.name)
        self.root = self.outer / "project"
        self.root.mkdir()
        self._root_patch = mock.patch.object(permissions, "PROJECT_ROOT", self.root)
        self._root_patch.start()

    def tearDown(self) -> None:
        self._root_patch.stop()
        self._tmp.cleanup()

    def invoke(self, args: dict) -> dict:
        return json.loads(check_syntax.invoke(args))

    def test_detects_alias_and_uses_file_scope(self) -> None:
        (self.root / "mod.py").write_text("print('ok')\n", encoding="utf-8")
        with (
            mock.patch.object(syntax.shutil, "which", side_effect=lambda cmd: "/usr/bin/python" if cmd == "python" else None),
            mock.patch.object(syntax.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")),
        ):
            result = self.invoke({"path": "mod.py", "language": "py"})

        self.assertTrue(result["ok"])
        self.assertEqual(result["language"], "python")
        self.assertEqual(result["scope"], "file")

    def test_directory_detects_python_project_marker(self) -> None:
        (self.root / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
        with (
            mock.patch.object(syntax.shutil, "which", side_effect=lambda cmd: "/usr/bin/python" if cmd == "python" else None),
            mock.patch.object(syntax.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")),
        ):
            result = self.invoke({"path": "."})

        self.assertTrue(result["ok"])
        self.assertEqual(result["language"], "python")
        self.assertEqual(result["scope"], "project")
        self.assertEqual(result["command"], "python -m compileall -q .")

    def test_marker_search_does_not_escape_project_root(self) -> None:
        (self.outer / "pyproject.toml").write_text("[project]\nname = 'outside'\n", encoding="utf-8")
        result = self.invoke({"path": "."})

        self.assertFalse(result["ok"])
        self.assertIn("No checker available", result["error"])

    def test_missing_path_returns_json_error(self) -> None:
        result = self.invoke({"path": "missing.py"})

        self.assertFalse(result["ok"])
        self.assertIn("does not exist", result["error"])
        self.assertEqual(result["errors"], [])

    def test_python_file_falls_back_to_py_compile_without_ruff(self) -> None:
        (self.root / "mod.py").write_text("print('ok')\n", encoding="utf-8")
        with (
            mock.patch.object(syntax.shutil, "which", side_effect=lambda cmd: "/usr/bin/python" if cmd == "python" else None),
            mock.patch.object(syntax.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")),
        ):
            result = self.invoke({"path": "mod.py"})

        self.assertTrue(result["ok"])
        self.assertIn("python -m py_compile", result["command"])

    def test_typescript_project_never_uses_npx(self) -> None:
        (self.root / "tsconfig.json").write_text("{}", encoding="utf-8")
        local_bin = self.root / "node_modules" / ".bin"
        local_bin.mkdir(parents=True)
        local_tsc = local_bin / "tsc"
        local_tsc.write_text("#!/bin/sh\n", encoding="utf-8")

        with mock.patch.object(syntax.subprocess, "run", return_value=mock.Mock(returncode=0, stdout="", stderr="")):
            result = self.invoke({"path": ".", "language": "typescript"})

        self.assertTrue(result["ok"])
        self.assertNotIn("npx", result["command"])
        self.assertIn(str(local_tsc), result["command"])
        self.assertIn("--noEmit", result["command"])

    def test_invalid_python_file_returns_parsed_line_error(self) -> None:
        if not shutil.which("python"):
            self.skipTest("python executable is not available")

        (self.root / "bad.py").write_text("def broken(:\n", encoding="utf-8")
        real_which = shutil.which

        def which_without_ruff(cmd: str) -> str | None:
            if cmd == "ruff":
                return None
            return real_which(cmd)

        with mock.patch.object(syntax.shutil, "which", side_effect=which_without_ruff):
            result = self.invoke({"path": "bad.py"})

        self.assertFalse(result["ok"])
        self.assertTrue(result["errors"])
        self.assertEqual(result["errors"][0]["file"], "bad.py")
        self.assertEqual(result["errors"][0]["line"], 1)
        self.assertIn("SyntaxError", result["errors"][0]["message"])


if __name__ == "__main__":
    unittest.main()
