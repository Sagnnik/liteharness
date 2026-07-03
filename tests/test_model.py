import os
import unittest
from argparse import Namespace
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

from config import settings
import model


class ModelFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        model.configure_model(None)

    def tearDown(self) -> None:
        model.configure_model(None)

    @mock.patch("model.ChatOpenRouter")
    def test_create_model_uses_settings(self, chat_openrouter) -> None:
        model.create_model("thread-1")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.model_name)
        self.assertEqual(kwargs["api_key"], settings.openai_api_key)
        self.assertEqual(kwargs["session_id"], "thread-1")

    @mock.patch("model.ChatOpenRouter")
    def test_create_model_applies_cli_overrides(self, chat_openrouter) -> None:
        model.configure_model(model.ModelOverrides(model_name="cli-model", openai_api_key="cli-key"))
        model.create_model("thread-2")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], "cli-model")
        self.assertEqual(kwargs["api_key"], "cli-key")
        self.assertEqual(kwargs["session_id"], "thread-2")

    @mock.patch("model.ChatOpenRouter")
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

    @mock.patch("model.ChatOpenRouter")
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

    @mock.patch("model.ChatOpenRouter")
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

    @mock.patch("model.ChatOpenRouter")
    def test_create_reflection_model_uses_reflection_name(self, chat_openrouter) -> None:
        model.create_reflection_model("thread-3")

        kwargs = chat_openrouter.call_args.kwargs
        self.assertEqual(kwargs["model"], settings.reflection_model_name)
        self.assertEqual(kwargs["api_key"], settings.openai_api_key)
        self.assertEqual(kwargs["session_id"], "thread-3:reflection")

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
