"""Terminal-safe rendering helpers for the Ness CLI adapter."""

from __future__ import annotations


def terminal_safe_text(value: object, *, multiline: bool = False) -> str:
    """Escape terminal controls in untrusted text while preserving readable Unicode."""
    result: list[str] = []
    for character in str(value):
        if multiline and character == "\n":
            result.append(character)
            continue
        if character.isprintable():
            result.append(character)
            continue
        codepoint = ord(character)
        if codepoint <= 0xFF:
            result.append(f"\\x{codepoint:02x}")
        elif codepoint <= 0xFFFF:
            result.append(f"\\u{codepoint:04x}")
        else:
            result.append(f"\\U{codepoint:08x}")
    return "".join(result)
