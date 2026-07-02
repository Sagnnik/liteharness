from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TranscriptLine:
    style: str
    text: str
    fragments: list[tuple[str, str]] | None = None
    # Set on the first row of a user block so it can be re-laid out on resize.
    user_source: str | None = None


@dataclass(frozen=True)
class MenuItem:
    key: str
    label: str
    description: str = ""
    suffix: str = ""
