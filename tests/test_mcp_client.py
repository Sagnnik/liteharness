"""Smoke tests for the legacy root mcp_client.MCPManager.

Live echo/filesystem subprocess integration was removed — it was flaky
(Connection closed / npx failures) and does not cover the SDK path
(``liteharness.mcp`` + ToolRegistry).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mcp_client import MCPManager


def test_startup_summary_when_unconfigured() -> None:
    mgr = MCPManager()
    message, level = mgr.startup_summary()
    assert level == "none"
    assert "none configured" in message
