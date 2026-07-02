from __future__ import annotations

import unittest

from reflection import ReflectionStructuredOutput, _normalize_bullets


class ReflectionHelperTests(unittest.TestCase):
    def test_normalize_bullets_caps_and_dedupes(self) -> None:
        bullets = _normalize_bullets(["- One", "One", "Two", "Three"])
        self.assertEqual(bullets, ["One", "Two"])

    def test_reflection_structured_output_defaults(self) -> None:
        output = ReflectionStructuredOutput()
        self.assertEqual(output.new_bullet_points, [])


if __name__ == "__main__":
    unittest.main()
