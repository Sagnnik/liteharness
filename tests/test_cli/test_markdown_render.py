from __future__ import annotations

from liteharness_cli.tui.markdown import markdown_transcript_lines
from liteharness_cli.tui.models import TranscriptLine


def _all_text(lines: list[TranscriptLine]) -> str:
    return "\n".join(line.text for line in lines)


def _assert_fragments_join(lines: list[TranscriptLine]) -> None:
    for line in lines:
        if line.fragments is None:
            continue
        joined = "".join(text for _, text in line.fragments)
        assert joined == line.text, f"fragment mismatch: {joined!r} != {line.text!r}"


def test_empty_returns_single_blank_line():
    lines = markdown_transcript_lines("", width=80)
    assert len(lines) == 1
    assert lines[0].text == ""


def test_complex_document_invariant_holds():
    text = (
        "# Heading\n\n"
        "Paragraph with **bold**, *italic*, `code` and [link](http://x).\n"
        "second line of paragraph.\n\n"
        "## Subheading\n\n"
        "- item one\n- item two\n\n"
        "1. first\n2. second\n\n"
        "```python\nx = 1\nprint(x)\n```\n\n"
        "> a quote\n> more quote\n\n"
        "---\n\n"
        "final plain text"
    )
    lines = markdown_transcript_lines(text, width=100)
    _assert_fragments_join(lines)
    joined = _all_text(lines)
    assert "http" not in joined
    for needle in ("Heading", "bold", "italic", "code", "link", "item one", "print", "final plain text"):
        assert needle in joined
