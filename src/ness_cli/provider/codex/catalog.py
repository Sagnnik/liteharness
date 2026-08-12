from __future__ import annotations

from ness_cli.provider.base import ModelInfo
from ness_cli.provider.codex.app_server import CodexAppServer


async def load_models(server: CodexAppServer) -> tuple[ModelInfo, ...]:
    records: list[ModelInfo] = []
    cursor: str | None = None
    while True:
        params = {"limit": 100}
        if cursor:
            params["cursor"] = cursor
        response = await server.request("model/list", params)
        for item in response.get("data") or []:
            if not isinstance(item, dict) or item.get("hidden"):
                continue
            efforts = tuple(
                str(option.get("reasoningEffort"))
                for option in item.get("supportedReasoningEfforts") or []
                if isinstance(option, dict) and option.get("reasoningEffort")
            )
            records.append(
                ModelInfo(
                    id=str(item.get("model") or item.get("id") or ""),
                    name=str(item.get("displayName") or item.get("model") or ""),
                    default_reasoning_effort=item.get("defaultReasoningEffort"),
                    reasoning_efforts=efforts,
                    supports_vision="image" in (item.get("inputModalities") or []),
                    is_default=bool(item.get("isDefault")),
                )
            )
        cursor = response.get("nextCursor")
        if not cursor:
            return tuple(records)
