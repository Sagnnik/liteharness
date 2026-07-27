import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import permissions


class PermissionsShellMatcherTests(unittest.TestCase):
    def test_blocks_semicolon_chain_allow_bypass(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("ls*", "ls; rm -rf /"),
        )

    def test_blocks_newline_chain(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("ls*", "ls\nrm -rf /"),
        )

    def test_blocks_and_chain(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("ls*", "ls && rm -rf /"),
        )

    def test_blocks_output_redirect(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("echo*", 'echo "malicious" > ~/.bashrc'),
        )

    def test_blocks_append_redirect(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("cat*", "cat /etc/passwd >> ./exfil.txt"),
        )

    def test_blocks_input_redirect(self) -> None:
        self.assertFalse(
            permissions._shell_command_matches("wc*", "wc -l < input.txt"),
        )

    def test_preserves_git_status_match(self) -> None:
        self.assertTrue(
            permissions._shell_command_matches("git status*", "git status -s"),
        )

    def test_allows_quoted_semicolon(self) -> None:
        self.assertTrue(
            permissions._shell_command_matches(
                "git commit*",
                'git commit -m "fix; bug"',
            ),
        )

    def test_chained_command_falls_through_to_ask(self) -> None:
        with mock.patch.object(permissions, "_load", return_value=permissions.DEFAULT_RULES.copy()):
            decision, rule = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "echo; sudo rm"},
            )
        self.assertEqual(decision, "ask")
        self.assertEqual(rule, "*")

    def test_shell_wildcard_rule_still_uses_safe_command_matching(self) -> None:
        rules = {"allow": ["shell:*"], "deny": [], "ask": ["*"]}
        with mock.patch.object(permissions, "_load", return_value=rules):
            safe_decision, safe_rule = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "pwd"},
            )
            unsafe_decision, unsafe_rule = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "pwd; sudo rm"},
            )
            read_decision, read_rule = permissions.check_with_rule(
                "shell",
                {"action": "read", "job_id": "abc"},
            )
        self.assertEqual((safe_decision, safe_rule), ("allow", "shell:*"))
        self.assertEqual((unsafe_decision, unsafe_rule), ("ask", "*"))
        self.assertEqual((read_decision, read_rule), ("allow", "shell:*"))


class PermissionsSessionRuleTests(unittest.TestCase):
    def setUp(self) -> None:
        permissions.clear_session_rules()

    def tearDown(self) -> None:
        permissions.clear_session_rules()

    def test_session_allow_matches(self) -> None:
        empty_rules = {"allow": [], "deny": [], "ask": ["*"]}
        with mock.patch.object(permissions, "_load", return_value=empty_rules):
            permissions.persist_rule("shell:run:pytest*", "allow", scope="session")
            decision, rule = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "pytest tests/test_foo.py"},
            )
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "shell:run:pytest*")

    def test_persistent_deny_beats_session_allow(self) -> None:
        rules = {
            "allow": [],
            "deny": ["shell:run:sudo*"],
            "ask": ["*"],
        }
        with mock.patch.object(permissions, "_load", return_value=rules):
            permissions.persist_rule("shell:run:sudo*", "allow", scope="session")
            decision, rule = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "sudo apt update"},
            )
        self.assertEqual(decision, "deny")
        self.assertEqual(rule, "shell:run:sudo*")

    def test_clear_session_rules_removes_session_allow(self) -> None:
        empty_rules = {"allow": [], "deny": [], "ask": ["*"]}
        with mock.patch.object(permissions, "_load", return_value=empty_rules):
            permissions.persist_rule("shell:run:pytest*", "allow", scope="session")
            permissions.clear_session_rules()
            decision, _ = permissions.check_with_rule(
                "shell",
                {"action": "run", "command": "pytest tests/test_foo.py"},
            )
        self.assertEqual(decision, "ask")

    def test_persistent_allow_survives_session_clear(self) -> None:
        rules = {
            "allow": ["shell:run:ls*"],
            "deny": [],
            "ask": ["*"],
        }
        with mock.patch.object(permissions, "_load", return_value=rules):
            permissions.persist_rule("shell:run:pytest*", "allow", scope="session")
            permissions.clear_session_rules()
            decision, rule = permissions.check_with_rule("shell", {"action": "run", "command": "ls -la"})
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "shell:run:ls*")


class DefaultRuleForTests(unittest.TestCase):
    def test_git_status_generalizes_subcommand(self) -> None:
        rule = permissions.default_rule_for("shell", {"action": "run", "command": "git status -s"})
        self.assertEqual(rule, "shell:run:git status*")

    def test_missing_action_defaults_to_run(self) -> None:
        rule = permissions.default_rule_for("shell", {"command": "git status -s"})
        self.assertEqual(rule, "shell:run:git status*")
        key = permissions.pattern_key("shell", {"command": "git status --short"})
        self.assertEqual(key, "shell:run:git status --short")

    def test_pytest_generalizes_base_command(self) -> None:
        rule = permissions.default_rule_for("shell", {"action": "run", "command": "pytest tests/foo.py"})
        self.assertEqual(rule, "shell:run:pytest*")

    def test_python_module_generalizes(self) -> None:
        rule = permissions.default_rule_for("shell", {"action": "run", "command": "python -m pytest tests/"})
        self.assertEqual(rule, "shell:run:python -m pytest*")

    def test_npm_run_generalizes(self) -> None:
        rule = permissions.default_rule_for("shell", {"action": "run", "command": "npm run build"})
        self.assertEqual(rule, "shell:run:npm run build*")

    def test_web_search_allowed_by_default(self) -> None:
        with mock.patch.object(permissions, "_load", return_value=permissions.DEFAULT_RULES.copy()):
            decision, rule = permissions.check_with_rule("web_search", {"query": "python docs"})
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "web_search:*")

    def test_webfetch_asks_by_default(self) -> None:
        with mock.patch.object(permissions, "_load", return_value=permissions.DEFAULT_RULES.copy()):
            decision, rule = permissions.check_with_rule("webfetch", {"url": "https://example.com/a"})
        self.assertEqual(decision, "ask")
        self.assertEqual(rule, "*")

    def test_webfetch_default_rule_uses_normalized_url_only(self) -> None:
        rule = permissions.default_rule_for(
            "webfetch",
            {"url": "https://Example.com:443/a?x=1#frag", "max_characters": 500},
        )
        self.assertEqual(rule, "webfetch:url=https://example.com/a?x=1")

    def test_webfetch_approval_ignores_max_characters(self) -> None:
        rules = {
            "allow": ["webfetch:url=https://example.com/a?x=1"],
            "deny": [],
            "ask": ["*"],
        }
        with mock.patch.object(permissions, "_load", return_value=rules):
            decision, rule = permissions.check_with_rule(
                "webfetch",
                {"url": "https://example.com/a?x=1", "max_characters": 5000},
            )
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "webfetch:url=https://example.com/a?x=1")

    def test_webfetch_different_query_still_asks(self) -> None:
        rules = {
            "allow": ["webfetch:url=https://example.com/a?x=1"],
            "deny": [],
            "ask": ["*"],
        }
        with mock.patch.object(permissions, "_load", return_value=rules):
            decision, rule = permissions.check_with_rule(
                "webfetch",
                {"url": "https://example.com/a?x=2"},
            )
        self.assertEqual(decision, "ask")
        self.assertEqual(rule, "*")

    def test_webfetch_wildcard_rule_does_not_bypass_per_url_approval(self) -> None:
        rules = {
            "allow": ["webfetch:*"],
            "deny": [],
            "ask": ["*"],
        }
        with mock.patch.object(permissions, "_load", return_value=rules):
            decision, rule = permissions.check_with_rule("webfetch", {"url": "https://example.com/a"})
        self.assertEqual(decision, "ask")
        self.assertEqual(rule, "*")

    def test_shell_read_allowed_by_default(self) -> None:
        with mock.patch.object(permissions, "_load", return_value=permissions.DEFAULT_RULES.copy()):
            decision, rule = permissions.check_with_rule("shell", {"action": "read", "job_id": "abc"})
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "shell:read:*")

    def test_shell_jobs_allowed_by_default(self) -> None:
        with mock.patch.object(permissions, "_load", return_value=permissions.DEFAULT_RULES.copy()):
            decision, rule = permissions.check_with_rule("shell", {"action": "jobs"})
        self.assertEqual(decision, "allow")
        self.assertEqual(rule, "shell:jobs:*")

    def test_non_shell_uses_exact_pattern_key(self) -> None:
        args = {"path": "foo.py", "content": "print('hi')"}
        self.assertEqual(
            permissions.default_rule_for("write", args),
            permissions.pattern_key("write", args),
        )


if __name__ == "__main__":
    unittest.main()
