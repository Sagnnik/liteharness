from __future__ import annotations

from permissions import PROJECT_ROOT

MENU_MAX_ROWS = 6
MENU_DESC_COL = 28
ESCAPE_KEY_FLUSH_TIMEOUT = 0
KEY_BINDING_TIMEOUT = 0.01
MOUSE_SCROLL_LINES = 3
FORM_FIELD_WIDTH = 32
PICKER_MODES = frozenset({"config_action", "config_models", "config_reasoning"})
FORM_LABELS = {
    "openai_key": "Provider API Key",
    "exa_key": "Exa API Key",
    "base_url": "Base URL",
}
ENV_PATH = PROJECT_ROOT / ".env"
