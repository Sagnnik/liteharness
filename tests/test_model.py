from __future__ import annotations

import os
import copy
import unittest
from argparse import Namespace
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

from ness_cli import chat_model as model
from ness_cli.config import settings


class ModelFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._previous_provider = settings.model_provider
        self._previous_profiles = copy.deepcopy(settings.provider_profiles)
        self._previous_api_key = settings.openai_api_key
        settings.model_provider = "openrouter"
        settings.provider_profiles["openrouter"] = {}
        settings.openai_api_key = "test"
        model.configure_model(None)

    def tearDown(self) -> None:
        model.configure_model(None)
        settings.provider_profiles = self._previous_profiles
        settings.openai_api_key = self._previous_api_key
        settings.model_provider = self._previous_provider

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_model_uses_settings(self, chat_openrouter) -> None:
        model.create_model("thread-1")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.model_name)
        self.assertEqual(kwargs["api_key"], settings.openai_api_key)
        self.assertEqual(kwargs["session_id"], "thread-1")

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_model_applies_cli_overrides(self, chat_openrouter) -> None:
        model.configure_model(model.ModelOverrides(model_name="cli-model", openai_api_key="cli-key"))
        model.create_model("thread-2")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], "cli-model")
        self.assertEqual(kwargs["api_key"], "cli-key")
        self.assertEqual(kwargs["session_id"], "thread-2")

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_model_passes_reasoning_effort(self, chat_openrouter) -> None:
        model.configure_model(
            model.ModelOverrides(
                model_name="deepseek/deepseek-v4-flash",
                reasoning_effort="high",
            )
        )
        model.create_model("thread-reasoning")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["reasoning"], {"effort": "high"})

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_model_omits_reasoning_for_non_reasoning_model(self, chat_openrouter) -> None:
        model.configure_model(
            model.ModelOverrides(
                model_name="openai/gpt-4o",
                reasoning_effort="high",
            )
        )
        model.create_model("thread-no-reasoning")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertNotIn("reasoning", kwargs)

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_model_omits_reasoning_when_effort_is_none(self, chat_openrouter) -> None:
        model.configure_model(
            model.ModelOverrides(
                model_name="deepseek/deepseek-v4-flash",
                reasoning_effort="none",
            )
        )
        model.create_model("thread-reasoning-off")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertNotIn("reasoning", kwargs)

    def test_set_active_reasoning_effort_rejects_unknown_value(self) -> None:
        model.set_active_model("deepseek/deepseek-v4-flash")
        with self.assertRaises(ValueError):
            model.set_active_reasoning_effort("extreme")  # type: ignore[arg-type]

    def test_set_active_model_coerces_invalid_effort(self) -> None:
        model.configure_model(model.ModelOverrides(reasoning_effort="medium"))
        coerced = model.set_active_model("deepseek/deepseek-v4-flash")
        self.assertEqual(coerced, "high")
        self.assertEqual(model.active_reasoning_effort(), "high")

    @mock.patch("ness_cli.chat_model.ChatOpenRouter")
    def test_create_reflection_model_uses_reflection_name(self, chat_openrouter) -> None:
        model.create_reflection_model("thread-3")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.reflection_model_name)
        self.assertEqual(kwargs["api_key"], settings.openai_api_key)
        self.assertEqual(kwargs["session_id"], "thread-3:reflection")

    def test_codex_reflection_uses_active_provider_model_by_default(self) -> None:
        previous_provider = settings.model_provider
        previous_profiles = copy.deepcopy(settings.provider_profiles)
        adapter = mock.Mock()
        adapter.build_chat_model.return_value = mock.Mock()
        try:
            settings.model_provider = "codex"
            settings.provider_profiles["codex"] = {"model_name": "gpt-codex"}
            with mock.patch("ness_cli.chat_model.active_provider", return_value=adapter):
                model.create_reflection_model("thread-codex")
        finally:
            settings.model_provider = previous_provider
            settings.provider_profiles = previous_profiles

        self.assertEqual(adapter.build_chat_model.call_args.kwargs["model_name"], "gpt-codex")

    def test_environment_model_settings_override_provider_profile(self) -> None:
        previous_model = settings.model_name
        previous_effort = settings.reasoning_effort
        settings.provider_profiles["openrouter"] = {
            "model_name": "profile-model",
            "reasoning_effort": "high",
        }
        try:
            settings.model_name = "environment-model"
            settings.reasoning_effort = "low"
            with mock.patch.dict(
                os.environ,
                {"MODEL_NAME": "environment-model", "REASONING_EFFORT": "low"},
            ):
                self.assertEqual(model.active_model_name(), "environment-model")
                self.assertEqual(model.active_reasoning_effort(), "low")
        finally:
            settings.model_name = previous_model
            settings.reasoning_effort = previous_effort

    def test_model_overrides_from_args_only_includes_set_values(self) -> None:
        overrides = model.model_overrides_from_args(
            Namespace(
                model="gpt-4o",
                reflection_model=None,
                api_key=None,
                base_url=None,
                openrouter_session_id=None,
            )
        )

        self.assertEqual(overrides, model.ModelOverrides(model_name="gpt-4o"))


if __name__ == "__main__":
    unittest.main()
