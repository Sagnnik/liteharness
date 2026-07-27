from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest import mock

from pydantic import BaseModel

os.environ.setdefault("OPENAI_API_KEY", "test")

from liteharness.permissions import DEFAULT_RULES, PermissionStore
from liteharness.tools.fs import (
    delete,
    edit,
    glob,
    read,
    write,
)

from tests.sdk_fixtures import SessionContextTestMixin


def _tool_json_schema(tool: Any) -> dict[str, Any]:
    args_schema = cast(type[BaseModel], tool.args_schema)
    return args_schema.model_json_schema()


class DeleteFileTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name))

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_deletes_existing_file(self) -> None:
        target = self.root / "old_module.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = delete.invoke({"path": "old_module.py"})
        self.assertEqual(result, "Deleted old_module.py")
        self.assertFalse(target.exists())

    def test_refuses_directory(self) -> None:
        (self.root / "pkg").mkdir()
        result = delete.invoke({"path": "pkg"})
        self.assertIn("directory", result)
        self.assertTrue((self.root / "pkg").is_dir())

    def test_refuses_missing_file(self) -> None:
        result = delete.invoke({"path": "missing.txt"})
        self.assertIn("does not exist", result)

    def test_refuses_git_paths(self) -> None:
        git_file = self.root / ".git" / "config"
        git_file.parent.mkdir(parents=True)
        git_file.write_text("[core]\n", encoding="utf-8")
        result = delete.invoke({"path": ".git/config"})
        self.assertIn("protected", result)
        self.assertTrue(git_file.exists())

    def test_refuses_ness_paths(self) -> None:
        ness = self.root / ".ness"
        ness.mkdir(exist_ok=True)
        targets = {
            "permissions.json": "{}",
            "hooks.json": "{}",
            "skills/demo/SKILL.md": "# Demo\n",
        }
        for rel, content in targets.items():
            path = ness / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")
            result = delete.invoke({"path": f".ness/{rel}"})
            with self.subTest(path=rel):
                self.assertIn("protected", result)
                self.assertTrue(path.exists())

    def test_refuses_path_outside_project(self) -> None:
        outside_parent = Path(tempfile.mkdtemp())
        outside = outside_parent / "outside_delete_test.txt"
        outside.write_text("nope", encoding="utf-8")
        try:
            result = delete.invoke({"path": str(outside)})
            self.assertTrue(result.startswith("Error:"))
            self.assertTrue(outside.exists())
        finally:
            outside.unlink(missing_ok=True)
            outside_parent.rmdir()


class ReadFileTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name))

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_default_limit_truncates_large_files(self) -> None:
        target = self.root / "large.txt"
        target.write_text("\n".join(f"line {idx}" for idx in range(1, 451)), encoding="utf-8")
        result = read.invoke({"path": "large.txt"})
        self.assertIn(" 400| line 400", result)
        self.assertNotIn(" 401| line 401", result)
        self.assertIn("truncated", result)

    def test_requested_limit_is_capped(self) -> None:
        target = self.root / "huge.txt"
        target.write_text("\n".join(f"line {idx}" for idx in range(1, 2501)), encoding="utf-8")
        result = read.invoke({"path": "huge.txt", "limit": 999999})
        self.assertIn("2000| line 2000", result)
        self.assertNotIn("2001| line 2001", result)
        self.assertIn("truncated", result)


class WriteFileTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name), format_on_write=False)

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_writes_normal_file(self) -> None:
        result = write.invoke({"path": "module.py", "content": "x = 1\n"})
        self.assertIn("Wrote 6 chars to module.py", result)
        self.assertEqual((self.root / "module.py").read_text(encoding="utf-8"), "x = 1\n")

    def test_refuses_protected_paths(self) -> None:
        for rel in (".git/config", ".ness/NESS.md"):
            with self.subTest(path=rel):
                result = write.invoke({"path": rel, "content": "blocked\n"})
                self.assertIn("protected", result)
                self.assertFalse((self.root / rel).exists())


class EditTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name), format_on_write=False)

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_applies_single_edit(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\n", encoding="utf-8")
        result = edit.invoke(
            {"path": "module.py", "old_string": "alpha", "new_string": "beta"}
        )
        self.assertIn("Applied 1 edit", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "beta\n")

    def test_sequential_edits_via_multiple_calls(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
        first = edit.invoke(
            {"path": "module.py", "old_string": "alpha", "new_string": "ALPHA"}
        )
        second = edit.invoke(
            {"path": "module.py", "old_string": "gamma", "new_string": "GAMMA"}
        )
        self.assertIn("Applied 1 edit", first)
        self.assertIn("Applied 1 edit", second)
        self.assertEqual(target.read_text(encoding="utf-8"), "ALPHA\nbeta\nGAMMA\n")

    def test_missing_old_string_fails_schema(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\n", encoding="utf-8")
        with self.assertRaises(Exception):
            edit.invoke({"path": "module.py", "new_string": "beta"})
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\n")

    def test_no_match_leaves_file_unchanged(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\nbeta\n", encoding="utf-8")
        result = edit.invoke(
            {
                "path": "module.py",
                "old_string": "nonexistent",
                "new_string": "X",
            }
        )
        self.assertIn("Error: no match for edit", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "alpha\nbeta\n")

    def test_schema_requires_old_and_new_string(self) -> None:
        schema = _tool_json_schema(edit)
        self.assertEqual(
            set(schema.get("required", [])),
            {"path", "old_string", "new_string"},
        )
        self.assertNotIn("edits", schema.get("properties", {}))

    def test_refuses_protected_paths(self) -> None:
        for rel in (".git/config", ".ness/NESS.md"):
            with self.subTest(path=rel):
                target = self.root / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("alpha\n", encoding="utf-8")
                result = edit.invoke(
                    {"path": rel, "old_string": "alpha", "new_string": "beta"}
                )
                self.assertIn("protected", result)
                self.assertEqual(target.read_text(encoding="utf-8"), "alpha\n")

    def test_fuzzy_match_below_threshold_leaves_file_unchanged(self) -> None:
        target = self.root / "module.py"
        target.write_text("def handle_request(request, response):\n    return response\n", encoding="utf-8")
        result = edit.invoke(
            {
                "path": "module.py",
                "old_string": "def handle_request(req, resp):\n    return resp\n",
                "new_string": "def handle_request(request, response):\n    return response\n",
            }
        )
        self.assertIn("Error: no match for edit", result)
        self.assertEqual(
            target.read_text(encoding="utf-8"),
            "def handle_request(request, response):\n    return response\n",
        )

    def test_fuzzy_match_above_threshold_emits_loud_warning(self) -> None:
        target = self.root / "module.py"
        original = "def process_payment(amount, currency, customer):\n    pass\n"
        target.write_text(original, encoding="utf-8")
        result = edit.invoke(
            {
                "path": "module.py",
                "old_string": "def process_payment(amount, currency, custemer):\n    pass\n",
                "new_string": "def process_payment(amount, currency, customer):\n    return amount\n",
            }
        )
        self.assertIn("WARNING: FUZZY MATCH", result)
        self.assertIn("verify the result before continuing", result)
        self.assertIn("Applied 1 edit", result)
        self.assertIn("return amount", target.read_text(encoding="utf-8"))

    def test_exact_match_does_not_warn_about_fuzzy(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\n", encoding="utf-8")
        result = edit.invoke(
            {"path": "module.py", "old_string": "alpha", "new_string": "beta"}
        )
        self.assertNotIn("FUZZY", result)
        self.assertIn("Applied 1 edit", result)

    def test_ambiguous_old_string_leaves_file_unchanged(self) -> None:
        target = self.root / "module.py"
        original = "alpha\nbeta\nalpha\n"
        target.write_text(original, encoding="utf-8")
        result = edit.invoke(
            {"path": "module.py", "old_string": "alpha", "new_string": "ALPHA"}
        )
        self.assertIn("found 2 matches for old_string", result)
        self.assertIn("replace_all=True", result)
        self.assertEqual(target.read_text(encoding="utf-8"), original)

    def test_replace_all_updates_every_match(self) -> None:
        target = self.root / "module.py"
        target.write_text("alpha\nbeta\nalpha\n", encoding="utf-8")
        result = edit.invoke(
            {
                "path": "module.py",
                "old_string": "alpha",
                "new_string": "ALPHA",
                "replace_all": True,
            }
        )
        self.assertIn("Applied 1 edit (2 replacements)", result)
        self.assertEqual(target.read_text(encoding="utf-8"), "ALPHA\nbeta\nALPHA\n")


class GlobFilesTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name))

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_finds_files_without_git_repo(self) -> None:
        (self.root / "src").mkdir()
        (self.root / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
        (self.root / "notes.txt").write_text("draft\n", encoding="utf-8")
        result = glob.invoke({"pattern": "**/*.py"})
        self.assertIn("src/app.py", result)
        self.assertNotIn("notes.txt", result)

    def test_finds_untracked_files_in_git_repo(self) -> None:
        subprocess.run(["git", "init"], cwd=self.root, capture_output=True, check=True)
        target = self.root / "scratch.py"
        target.write_text("x = 1\n", encoding="utf-8")
        result = glob.invoke({"pattern": "scratch.py"})
        self.assertIn("scratch.py", result)


class DeleteFilePermissionTests(unittest.TestCase):
    def test_shell_run_rm_denied_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ness = root / ".ness"
            ness.mkdir()
            store = PermissionStore(ness_dir=ness, project_root=root)
            with mock.patch.object(store, "_load", return_value=DEFAULT_RULES.copy()):
                decision, rule = store.check_with_rule(
                    "shell", {"action": "run", "command": "rm old_file.py"}
                )
        self.assertEqual(decision, "deny")
        self.assertEqual(rule, "shell:run:rm*")

    def test_shell_start_rm_rf_still_denied(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ness = root / ".ness"
            ness.mkdir()
            store = PermissionStore(ness_dir=ness, project_root=root)
            with mock.patch.object(store, "_load", return_value=DEFAULT_RULES.copy()):
                decision, rule = store.check_with_rule(
                    "shell", {"action": "start", "command": "rm -rf build/"}
                )
        self.assertEqual(decision, "deny")
        self.assertEqual(rule, "shell:start:rm*")


if __name__ == "__main__":
    unittest.main()
