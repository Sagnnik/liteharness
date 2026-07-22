"""LiteHarness CLI companion package (adapter tier).

Sits between the domain-agnostic SDK (:mod:`liteharness`) and the TUI
(``cli/``). Holds the coding-specific config, model factory, overlay,
prompts, rollback helpers, @file mention expansion, event reconstruction,
and the :class:`CodingSession` adapter wrapping :class:`liteharness.Session`.
"""

from liteharness_cli.coding_session import CodingSession

__all__ = ["CodingSession"]