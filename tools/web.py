from __future__ import annotations

import ipaddress
import json
import socket
import time
from typing import Any, Literal
from urllib.parse import urlparse, urlunparse

from langchain_core.tools import tool
import requests

from config import settings

_ALLOWED_URL_SCHEMES = frozenset({"http", "https"})
_BLOCKED_HOSTNAMES = frozenset(
    {
        "localhost",
        "0.0.0.0",
        "127.0.0.1",
        "::1",
        "169.254.169.254",
    }
)
_EXA_BASE_URL = "https://api.exa.ai"
_EXA_TIMEOUT = (3.0, 20.0)
_EXA_MAX_RETRIES = 2
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_MAX_RESULTS_MIN = 1
_MAX_RESULTS_MAX = 10
_MAX_CHARACTERS_MIN = 1
_MAX_CHARACTERS_MAX = 30_000


class UrlValidationError(ValueError):
    """Raised when a fetch URL fails local validation."""


class ExaRequestError(RuntimeError):
    """Raised when Exa returns an API error."""

    def __init__(self, message: str, category: str = "api") -> None:
        super().__init__(message)
        self.category = category


def _normalize_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in _ALLOWED_URL_SCHEMES:
        msg = f"URL scheme not allowed: {parsed.scheme!r} (must be http or https)"
        raise UrlValidationError(msg)

    hostname = parsed.hostname
    if not hostname:
        raise UrlValidationError("URL is missing a hostname")

    if parsed.username or parsed.password:
        raise UrlValidationError("URL userinfo is not allowed")

    lowered = hostname.lower().rstrip(".")
    if lowered in _BLOCKED_HOSTNAMES or lowered.endswith(".localhost"):
        raise UrlValidationError(f"URL hostname {hostname!r} is not allowed")

    try:
        ip = ipaddress.ip_address(lowered)
    except ValueError:
        _validate_resolved_host(lowered)
    else:
        _validate_ip_address(ip, hostname)

    netloc = lowered
    if ":" in lowered and not lowered.startswith("["):
        netloc = f"[{lowered}]"
    try:
        port = parsed.port
    except ValueError as exc:
        raise UrlValidationError("URL port is invalid") from exc
    if port and not _is_default_port(parsed.scheme, port):
        netloc = f"{netloc}:{port}"

    return urlunparse(
        (
            parsed.scheme.lower(),
            netloc,
            parsed.path or "",
            parsed.params or "",
            parsed.query or "",
            "",
        )
    )


def _validate_url(url: str) -> None:
    _normalize_url(url)


def _validate_resolved_host(hostname: str) -> None:
    try:
        records = socket.getaddrinfo(hostname, None, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UrlValidationError(f"URL hostname {hostname!r} could not be resolved") from exc
    if not records:
        raise UrlValidationError(f"URL hostname {hostname!r} could not be resolved")

    for record in records:
        sockaddr = record[4]
        if not sockaddr:
            continue
        _validate_ip_address(ipaddress.ip_address(sockaddr[0]), hostname)


def _validate_ip_address(ip: ipaddress._BaseAddress, hostname: str) -> None:
    if (
        not ip.is_global
        or ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    ):
        raise UrlValidationError(f"URL hostname {hostname!r} resolves to blocked address {ip}")


def _is_default_port(scheme: str, port: int) -> bool:
    return (scheme == "http" and port == 80) or (scheme == "https" and port == 443)


def _json_response(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2)


def _missing_exa_error(**extra: Any) -> str:
    return _json_response(
        {
            "error": "Exa API key not configured. Please set EXA_API_KEY environment variable.",
            "category": "configuration",
            **extra,
        }
    )


def _result_content(result: dict[str, Any]) -> str:
    highlights = result.get("highlights")
    if highlights:
        return "\n".join(highlights)
    text = result.get("text")
    if text:
        return text
    summary = result.get("summary")
    if summary:
        return summary
    return ""


def _normalize_search_results(query: str, response: dict[str, Any]) -> dict[str, Any]:
    results = []
    for item in response.get("results") or []:
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "content": _result_content(item),
                "published_date": item.get("publishedDate") or item.get("published_date"),
            }
        )
    return {"query": query, "results": results}


def _exa_request(endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
    if not settings.has_exa:
        raise ExaRequestError("Exa API key not configured. Please set EXA_API_KEY environment variable.", "configuration")

    headers = {
        "x-api-key": settings.exa_api_key or "",
        "Content-Type": "application/json",
        "User-Agent": "liteharness-web-tools/1.0",
    }
    url = f"{_EXA_BASE_URL}{endpoint}"
    last_error: str | None = None

    for attempt in range(_EXA_MAX_RETRIES + 1):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=_EXA_TIMEOUT)
        except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
            last_error = _redact_error(str(exc))
            if attempt < _EXA_MAX_RETRIES:
                time.sleep(0.25 * (2**attempt))
                continue
            category = "timeout" if isinstance(exc, requests.Timeout) else "api"
            raise ExaRequestError(f"Exa request failed: {last_error}", category) from exc

        if response.status_code in _TRANSIENT_STATUS_CODES and attempt < _EXA_MAX_RETRIES:
            time.sleep(0.25 * (2**attempt))
            continue

        if response.status_code >= 400:
            detail = _response_error_detail(response)
            raise ExaRequestError(f"Exa request failed with status {response.status_code}: {detail}")

        try:
            data = response.json()
        except ValueError as exc:
            raise ExaRequestError("Exa returned invalid JSON") from exc
        if not isinstance(data, dict):
            raise ExaRequestError("Exa returned an unexpected response shape")
        return data

    raise ExaRequestError(f"Exa request failed: {last_error or 'unknown error'}")


def _response_error_detail(response: requests.Response) -> str:
    try:
        data = response.json()
    except ValueError:
        return _redact_error(response.text)
    if isinstance(data, dict):
        message = data.get("error") or data.get("message") or data.get("detail")
        if message:
            return _redact_error(str(message))
    return "provider error"


def _redact_error(message: str, max_length: int = 300) -> str:
    compact = " ".join(message.split())
    if settings.exa_api_key:
        compact = compact.replace(settings.exa_api_key, "[REDACTED]")
    if len(compact) > max_length:
        return compact[: max_length - 3] + "..."
    return compact


def _validate_search_args(query: str, max_results: int) -> str | None:
    if not query.strip():
        return "query must be non-empty"
    if not _MAX_RESULTS_MIN <= max_results <= _MAX_RESULTS_MAX:
        return f"max_results must be between {_MAX_RESULTS_MIN} and {_MAX_RESULTS_MAX}"
    return None


def _validate_fetch_args(max_characters: int) -> str | None:
    if not _MAX_CHARACTERS_MIN <= max_characters <= _MAX_CHARACTERS_MAX:
        return f"max_characters must be between {_MAX_CHARACTERS_MIN} and {_MAX_CHARACTERS_MAX}"
    return None


@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_type: Literal["auto", "instant", "deep-lite", "deep", "deep-reasoning"] = "auto",
    include_domains: list[str] | None = None,
) -> str:
    """Search the web using Exa for current information and documentation.

    After receiving results, synthesize the information into a helpful response and cite sources.
    Use include_domains to scope searches (e.g. ["docs.python.org", "github.com"]).
    """
    validation_error = _validate_search_args(query, max_results)
    if validation_error:
        return _json_response({"error": validation_error, "query": query, "category": "validation"})

    payload: dict[str, Any] = {
        "query": query,
        "numResults": max_results,
        "type": search_type,
        "contents": {"highlights": True},
    }
    if include_domains:
        payload["includeDomains"] = include_domains

    try:
        response = _exa_request("/search", payload)
        return _json_response(_normalize_search_results(query, response))
    except ExaRequestError as exc:
        if exc.category == "configuration":
            return _missing_exa_error(query=query)
        return _json_response({"error": f"Web search error: {exc}", "query": query, "category": exc.category})


@tool
def fetch_url(url: str, max_characters: int = 12000) -> str:
    """Fetch content from a URL as clean markdown via Exa.

    Use after web_search when you need full page content beyond search highlights.
    After receiving content, synthesize relevant information for the user.
    """
    validation_error = _validate_fetch_args(max_characters)
    if validation_error:
        return _json_response({"error": validation_error, "url": url, "category": "validation"})

    try:
        normalized_url = _normalize_url(url)
    except UrlValidationError as exc:
        return _json_response({"error": str(exc), "url": url, "category": "validation"})

    try:
        response = _exa_request(
            "/contents",
            {
                "urls": [normalized_url],
                "text": {"maxCharacters": max_characters, "includeHtmlTags": False},
            },
        )
    except ExaRequestError as exc:
        if exc.category == "configuration":
            return _missing_exa_error(url=normalized_url)
        return _json_response({"error": f"Fetch URL error: {exc}", "url": normalized_url, "category": exc.category})

    for status in response.get("statuses") or []:
        if status.get("status") == "error":
            return _json_response(
                {
                    "error": f"Exa could not fetch {status.get('id') or normalized_url}",
                    "url": normalized_url,
                    "category": "api",
                }
            )

    results = response.get("results") or []
    if not results:
        return _json_response({"error": "No content returned for URL", "url": normalized_url, "category": "api"})

    page = results[0]
    markdown_content = page.get("text") or ""
    if not markdown_content.strip():
        return _json_response({"error": "No markdown content returned for URL", "url": normalized_url, "category": "api"})

    return _json_response(
        {
            "url": page.get("url") or normalized_url,
            "title": page.get("title"),
            "markdown_content": markdown_content,
            "content_length": len(markdown_content),
        }
    )
