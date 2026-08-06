"""Transactional import of explicitly selected MCP JSON configurations."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from ness_cli.config_store import atomic_write_json, load_configs, write_config

_IMPORTS_KEY = "mcp_imports"
_PLACEHOLDER = re.compile(r"\$\{[^{}]+\}")


@dataclass(frozen=True)
class ImportEntry:
    name: str
    action: str
    summary: str
    entry: dict[str, Any]
    warnings: tuple[str, ...] = ()


@dataclass
class MCPImportPlan:
    source: Path
    destination: Path
    project_root: Path
    source_digest: str
    entries: list[ImportEntry] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    destination_document: dict[str, Any] | None = None
    destination_key: str = "mcpServers"

    @property
    def valid(self) -> bool:
        return not self.errors

    @property
    def changes(self) -> list[ImportEntry]:
        return [entry for entry in self.entries if entry.action in {"add", "replace"}]

    def render(self) -> str:
        lines = [f"Source: {self.source}", f"Destination: {self.destination}"]
        for entry in self.entries:
            lines.append(f"- {entry.name}: {entry.action} — {entry.summary}")
            lines.extend(f"  warning: {warning}" for warning in entry.warnings)
        lines.extend(f"Warning: {warning}" for warning in self.warnings)
        lines.extend(f"Error: {error}" for error in self.errors)
        return "\n".join(lines)


def plan_mcp_import(
    source: Path,
    destination: Path,
    *,
    project_root: Path,
    selected: set[str] | None = None,
    replace: set[str] | None = None,
) -> MCPImportPlan:
    source = source.expanduser().resolve()
    destination = destination.resolve()
    replace = set(replace or ())
    try:
        source_bytes = source.read_bytes()
        source_doc = json.loads(source_bytes)
    except OSError as exc:
        return MCPImportPlan(
            source, destination, project_root.resolve(), "", errors=[f"cannot read source: {exc}"]
        )
    except json.JSONDecodeError as exc:
        return MCPImportPlan(
            source,
            destination,
            project_root.resolve(),
            "",
            errors=[f"source contains invalid JSON at line {exc.lineno}, column {exc.colno}"],
        )

    plan = MCPImportPlan(
        source=source,
        destination=destination,
        project_root=project_root.resolve(),
        source_digest=hashlib.sha256(source_bytes).hexdigest(),
    )
    if not isinstance(source_doc, dict):
        plan.errors.append("source root must be a JSON object")
        return plan
    if "mcpServers" in source_doc and "servers" in source_doc:
        plan.warnings.append("source contains both keys; using mcpServers")
    source_servers = source_doc.get("mcpServers", source_doc.get("servers"))
    if not isinstance(source_servers, dict):
        plan.errors.append("source must contain an object-valued mcpServers or servers key")
        return plan

    if selected is not None:
        missing = sorted(selected - set(source_servers))
        if missing:
            plan.errors.append(f"source does not contain requested server(s): {', '.join(missing)}")
        names = [name for name in source_servers if name in selected]
    else:
        names = list(source_servers)

    destination_doc, destination_key, destination_error = _load_destination(destination)
    if destination_error:
        plan.errors.append(destination_error)
        return plan
    plan.destination_document = destination_doc
    plan.destination_key = destination_key
    existing = destination_doc[destination_key]
    assert isinstance(existing, dict)

    conflicts: set[str] = set()
    for name in names:
        raw = source_servers[name]
        if not isinstance(name, str) or not name.strip():
            plan.errors.append("source server names must be non-empty strings")
            continue
        errors, warnings = validate_import_entry(raw)
        if errors:
            plan.errors.extend(f"{name}: {error}" for error in errors)
            continue
        assert isinstance(raw, dict)
        if name not in existing:
            action = "add"
        elif _entry_digest(existing[name]) == _entry_digest(raw):
            action = "unchanged"
        else:
            action = "replace" if name in replace else "conflict"
            conflicts.add(name)
            if name not in replace:
                plan.errors.append(
                    f"{name}: destination differs; rerun with --replace {name}"
                )
        plan.entries.append(
            ImportEntry(
                name=name,
                action=action,
                summary=_entry_summary(raw),
                entry=dict(raw),
                warnings=tuple(warnings),
            )
        )

    invalid_replace = sorted(replace - conflicts)
    if invalid_replace:
        plan.errors.append(
            "--replace named server(s) that are not selected conflicts: "
            + ", ".join(invalid_replace)
        )
    return plan


def execute_mcp_import(plan: MCPImportPlan, *, config_dir: Path) -> list[str]:
    if not plan.valid or plan.destination_document is None:
        raise ValueError("cannot execute an invalid MCP import plan")
    if not plan.changes:
        return []
    document = dict(plan.destination_document)
    server_map = dict(document[plan.destination_key])
    for item in plan.changes:
        server_map[item.name] = item.entry
    document[plan.destination_key] = server_map
    atomic_write_json(plan.destination, document)

    warnings: list[str] = []
    try:
        configs = load_configs(config_dir)
        all_projects = configs.get(_IMPORTS_KEY, {})
        all_projects = dict(all_projects) if isinstance(all_projects, dict) else {}
        project_key = str(plan.project_root)
        project_entries = all_projects.get(project_key, {})
        project_entries = dict(project_entries) if isinstance(project_entries, dict) else {}
        now = datetime.now(timezone.utc).isoformat()
        for item in plan.changes:
            project_entries[item.name] = {
                "source_path": str(plan.source),
                "source_digest": plan.source_digest,
                "entry_digest": _entry_digest(item.entry),
                "imported_at": now,
            }
        all_projects[project_key] = project_entries
        write_config(_IMPORTS_KEY, all_projects, config_dir)
    except OSError as exc:
        warnings.append(f"config imported, but provenance could not be saved: {exc}")
    return warnings


def provenance_for_server(
    *,
    config_dir: Path,
    project_root: Path,
    name: str,
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    projects = load_configs(config_dir).get(_IMPORTS_KEY, {})
    if not isinstance(projects, dict):
        return None
    project = projects.get(str(project_root.resolve()), {})
    if not isinstance(project, dict):
        return None
    value = project.get(name)
    if not isinstance(value, dict):
        return None
    result = dict(value)
    result["modified"] = result.get("entry_digest") != _entry_digest(entry)
    return result


def validate_import_entry(value: Any) -> tuple[list[str], list[str]]:
    if not isinstance(value, dict):
        return ["server definition must be an object"], []
    errors: list[str] = []
    warnings: list[str] = []
    command = value.get("command")
    url = value.get("url")
    raw_type = value.get("type")
    if raw_type is not None and not isinstance(raw_type, str):
        errors.append("type must be a string")
    transport = raw_type.lower() if isinstance(raw_type, str) else ("http" if url is not None else "stdio" if command is not None else "")
    if transport == "streamable-http":
        transport = "http"
    if transport in {"sse", "ws", "websocket"}:
        errors.append(f"unsupported transport: {transport}")
    elif transport not in {"stdio", "http"}:
        errors.append("type must be stdio, http, or streamable-http")
    if command is not None and url is not None:
        errors.append("server cannot contain both command and url")
    description = value.get("description", "")
    if not isinstance(description, str):
        errors.append("description must be a string")
    startup_timeout = value.get("startup_timeout", 20)
    if (
        isinstance(startup_timeout, bool)
        or not isinstance(startup_timeout, (int, float))
        or startup_timeout <= 0
    ):
        errors.append("startup_timeout must be a positive number")
    if transport == "stdio":
        if not (
            isinstance(command, str) and bool(command.strip())
            or isinstance(command, list)
            and bool(command)
            and all(isinstance(part, str) and part for part in command)
        ):
            errors.append("stdio server requires a non-empty command")
        args = value.get("args", [])
        if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
            errors.append("args must be an array of strings")
        for field_name in ("cwd", "envFile"):
            if value.get(field_name) is not None and not isinstance(value[field_name], str):
                errors.append(f"{field_name} must be a string")
    if transport == "http":
        if not isinstance(url, str) or not url:
            errors.append("http server requires a non-empty url")
        elif not _PLACEHOLDER.search(url):
            try:
                parts = urlsplit(url)
                _ = parts.port
                if parts.scheme not in {"http", "https"} or not parts.hostname:
                    errors.append("url must be an http(s) URL with a hostname")
            except ValueError:
                errors.append("url is malformed")
        if value.get("envFile") is not None:
            errors.append("envFile is only supported for stdio servers")
    for field_name in ("env", "headers"):
        field_value = value.get(field_name, {})
        if not isinstance(field_value, dict) or not all(
            isinstance(key, str) and isinstance(item, str)
            for key, item in field_value.items()
        ):
            errors.append(f"{field_name} must be an object of string values")
    if value.get("headersHelper") is not None:
        errors.append("headersHelper is not supported")
    if value.get("auth") is not None and value.get("oauth") is not None:
        errors.append("server cannot contain both auth and oauth")
    _validate_import_oauth(value, errors)
    headers = value.get("headers", {})
    if (
        value.get("auth") is not None or value.get("oauth") is not None
    ) and isinstance(headers, dict) and any(
        isinstance(key, str) and key.lower() == "authorization" for key in headers
    ):
        errors.append("OAuth cannot be combined with an explicit Authorization header")
    if transport != "http" and (
        value.get("auth") is not None or value.get("oauth") is not None
    ):
        errors.append("OAuth is supported only for HTTP servers")
    if _contains_placeholder(value):
        warnings.append("contains unresolved placeholders; runtime environment must provide them")
    if _contains_literal_secret(value):
        warnings.append("contains literal credential values; environment placeholders are safer")
    return errors, warnings


def _validate_import_oauth(value: dict[str, Any], errors: list[str]) -> None:
    auth = value.get("auth")
    oauth = value.get("oauth")
    if auth is not None:
        if not isinstance(auth, dict):
            errors.append("auth must be an object")
        elif not isinstance(auth.get("CLIENT_ID"), str) or not auth.get("CLIENT_ID"):
            errors.append("auth.CLIENT_ID must be a non-empty string")
        else:
            secret = auth.get("CLIENT_SECRET")
            if secret is not None and not isinstance(secret, str):
                errors.append("auth.CLIENT_SECRET must be a string")
            scopes = auth.get("scopes", [])
            if not (
                isinstance(scopes, str)
                or isinstance(scopes, list)
                and all(isinstance(scope, str) for scope in scopes)
            ):
                errors.append("auth.scopes must be a string or array of strings")
    if oauth is not None:
        if not isinstance(oauth, dict):
            errors.append("oauth must be an object")
            return
        if oauth.get("authServerMetadataUrl") is not None:
            errors.append("oauth.authServerMetadataUrl is not supported")
        client_id = oauth.get("clientId")
        if client_id is not None and not isinstance(client_id, str):
            errors.append("oauth.clientId must be a string")
        client_secret = oauth.get("clientSecret")
        if client_secret is not None and not isinstance(client_secret, str):
            errors.append("oauth.clientSecret must be a string")
        port = oauth.get("callbackPort")
        if port is not None and (
            isinstance(port, bool) or not isinstance(port, int) or not 1 <= port <= 65535
        ):
            errors.append("oauth.callbackPort must be an integer from 1 to 65535")
        scopes = oauth.get("scopes", [])
        if not (
            isinstance(scopes, str)
            or isinstance(scopes, list)
            and all(isinstance(scope, str) for scope in scopes)
        ):
            errors.append("oauth.scopes must be a string or array of strings")
        method = oauth.get("tokenEndpointAuthMethod")
        if method is not None and method not in {
            "none",
            "client_secret_post",
            "client_secret_basic",
        }:
            errors.append(
                "oauth.tokenEndpointAuthMethod must be none, client_secret_post, or client_secret_basic"
            )


def _load_destination(path: Path) -> tuple[dict[str, Any], str, str | None]:
    if not path.exists():
        return {"mcpServers": {}}, "mcpServers", None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        return {}, "mcpServers", f"cannot read destination: {exc}"
    except json.JSONDecodeError as exc:
        return {}, "mcpServers", f"destination contains invalid JSON at line {exc.lineno}, column {exc.colno}"
    if not isinstance(value, dict):
        return {}, "mcpServers", "destination root must be a JSON object"
    if "servers" in value:
        return {}, "mcpServers", "destination uses unsupported 'servers'; migrate it to 'mcpServers' before importing"
    if "mcpServers" not in value:
        value["mcpServers"] = {}
    if not isinstance(value["mcpServers"], dict):
        return {}, "mcpServers", "destination mcpServers must be a JSON object"
    return value, "mcpServers", None


def _entry_summary(value: dict[str, Any]) -> str:
    if isinstance(value.get("url"), str):
        target = _safe_url(value["url"])
        mode = "oauth" if value.get("auth") is not None or value.get("oauth") is not None else "headers"
        return f"http {target} ({mode})"
    command = value.get("command")
    executable = command[0] if isinstance(command, list) and command else command
    return f"stdio {executable}"


def _safe_url(value: str) -> str:
    try:
        parts = urlsplit(value)
        host = parts.hostname or ""
        if parts.port:
            host = f"{host}:{parts.port}"
        return urlunsplit((parts.scheme, host, parts.path, "", ""))
    except ValueError:
        return "[invalid URL]"


def _entry_digest(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _contains_placeholder(value: Any) -> bool:
    if isinstance(value, str):
        return bool(_PLACEHOLDER.search(value))
    if isinstance(value, dict):
        return any(_contains_placeholder(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_placeholder(item) for item in value)
    return False


def _contains_literal_secret(value: dict[str, Any]) -> bool:
    candidates: list[str] = []
    for field_name in ("env", "headers"):
        mapping = value.get(field_name)
        if isinstance(mapping, dict):
            candidates.extend(item for item in mapping.values() if isinstance(item, str))
    auth = value.get("auth")
    if isinstance(auth, dict) and isinstance(auth.get("CLIENT_SECRET"), str):
        candidates.append(auth["CLIENT_SECRET"])
    oauth = value.get("oauth")
    if isinstance(oauth, dict) and isinstance(oauth.get("clientSecret"), str):
        candidates.append(oauth["clientSecret"])
    return any(candidate and not _PLACEHOLDER.search(candidate) for candidate in candidates)
