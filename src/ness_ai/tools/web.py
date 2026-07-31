from __future__ import annotations

import ipaddress
import json
import re
import socket
import time
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.parse import parse_qs, unquote, urlparse, urlunparse

from langchain_core.tools import tool
import requests

from ness_ai.session_context import try_get_session_context

try:
    from bs4 import BeautifulSoup

    _HAS_BS4 = True
except ImportError:
    _HAS_BS4 = False

try:
    import trafilatura

    _HAS_TRAFILATURA = True
except ImportError:
    _HAS_TRAFILATURA = False


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
_TRANSIENT_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

_EXA_BASE_URL = "https://api.exa.ai"
_EXA_TIMEOUT = (3.0, 20.0)
_EXA_MAX_RETRIES = 2

_DD_SEARCH_URL = "https://html.duckduckgo.com/html/"
_DD_TIMEOUT = (5.0, 20.0)
_DD_MAX_RETRIES = 2
_FETCH_TIMEOUT = (5.0, 20.0)
_FETCH_MAX_RETRIES = 1
_FETCH_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

_MAX_RESULTS_MIN = 1
_MAX_RESULTS_MAX = 10
_MAX_CHARACTERS_MIN = 1
_MAX_CHARACTERS_MAX = 30_000


class UrlValidationError(ValueError):
    """Raised when a fetch URL fails local validation."""


class ProviderError(RuntimeError):
    """Raised when any search provider returns an error."""

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


def _exa_api_key() -> str | None:
    rt = try_get_session_context()
    if rt is None:
        return None
    return rt.options.exa_api_key


def _has_exa() -> bool:
    return bool(_exa_api_key())


def _redact_error(message: str, max_length: int = 300) -> str:
    compact = " ".join(message.split())
    key = _exa_api_key()
    if key:
        compact = compact.replace(key, "[REDACTED]")
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


def _domain_matches(hostname: str, domain: str) -> bool:
    hostname = hostname.lower().lstrip(".")
    domain = domain.lower().lstrip(".")
    return hostname == domain or hostname.endswith(f".{domain}")


def _retry_transient(
    func: Any,
    max_retries: int,
    transient_codes: frozenset[int] = _TRANSIENT_STATUS_CODES,
) -> requests.Response:
    """Execute a requests call with retry on transient failures."""
    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = func()
        except (requests.Timeout, requests.ConnectionError) as exc:
            last_exc = exc
            if attempt < max_retries:
                time.sleep(0.5 * (2**attempt))
                continue
            raise
        if response.status_code in transient_codes and attempt < max_retries:
            time.sleep(0.5 * (2**attempt))
            continue
        return response
    raise last_exc  # type: ignore[misc]


@runtime_checkable
class SearchProvider(Protocol):
    """Interface for web search/fetch providers."""

    def search(
        self,
        query: str,
        max_results: int,
        search_type: str,
        include_domains: list[str] | None,
    ) -> dict[str, Any]:
        ...

    def fetch(self, url: str, max_characters: int) -> dict[str, Any]:
        ...


class ExaProvider:
    """Exa AI search provider — requires EXA_API_KEY."""

    def search(
        self,
        query: str,
        max_results: int,
        search_type: str,
        include_domains: list[str] | None,
    ) -> dict[str, Any]:
        if not _has_exa():
            raise ProviderError(
                "Exa API key not configured. Please set EXA_API_KEY environment variable.",
                "configuration",
            )

        payload: dict[str, Any] = {
            "query": query,
            "numResults": max_results,
            "type": search_type,
            "contents": {"highlights": True},
        }
        if include_domains:
            payload["includeDomains"] = include_domains

        response = self._request("/search", payload)
        results = []
        for item in response.get("results") or []:
            results.append(
                {
                    "title": item.get("title"),
                    "url": item.get("url"),
                    "content": self._result_content(item),
                    "published_date": item.get("publishedDate") or item.get("published_date"),
                }
            )
        return {"query": query, "results": results}

    def fetch(self, url: str, max_characters: int) -> dict[str, Any]:
        if not _has_exa():
            raise ProviderError(
                "Exa API key not configured. Please set EXA_API_KEY environment variable.",
                "configuration",
            )

        response = self._request(
            "/contents",
            {
                "urls": [url],
                "text": {"maxCharacters": max_characters, "includeHtmlTags": False},
            },
        )

        for status in response.get("statuses") or []:
            if status.get("status") == "error":
                raise ProviderError(
                    f"Exa could not fetch {status.get('id') or url}", "api"
                )

        results = response.get("results") or []
        if not results:
            raise ProviderError("No content returned for URL", "api")

        page = results[0]
        markdown_content = page.get("text") or ""
        if not markdown_content.strip():
            raise ProviderError("No markdown content returned for URL", "api")

        return {
            "url": page.get("url") or url,
            "title": page.get("title"),
            "markdown_content": markdown_content,
            "content_length": len(markdown_content),
        }

    def _request(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {
            "x-api-key": _exa_api_key() or "",
            "Content-Type": "application/json",
            "User-Agent": "ness_ai-web-tools/1.0",
        }
        url = f"{_EXA_BASE_URL}{endpoint}"
        last_error: str | None = None

        for attempt in range(_EXA_MAX_RETRIES + 1):
            try:
                response = requests.post(
                    url, headers=headers, json=payload, timeout=_EXA_TIMEOUT
                )
            except (requests.Timeout, requests.ConnectionError, requests.RequestException) as exc:
                last_error = _redact_error(str(exc))
                if attempt < _EXA_MAX_RETRIES:
                    time.sleep(0.25 * (2**attempt))
                    continue
                category = "timeout" if isinstance(exc, requests.Timeout) else "api"
                raise ProviderError(f"Exa request failed: {last_error}", category) from exc

            if response.status_code in _TRANSIENT_STATUS_CODES and attempt < _EXA_MAX_RETRIES:
                time.sleep(0.25 * (2**attempt))
                continue

            if response.status_code >= 400:
                detail = self._response_error_detail(response)
                raise ProviderError(
                    f"Exa request failed with status {response.status_code}: {detail}"
                )

            try:
                data = response.json()
            except ValueError as exc:
                raise ProviderError("Exa returned invalid JSON") from exc
            if not isinstance(data, dict):
                raise ProviderError("Exa returned an unexpected response shape")
            return data

        raise ProviderError(f"Exa request failed: {last_error or 'unknown error'}")

    @staticmethod
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

    @staticmethod
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


class DuckDuckGoProvider:
    """Keyless fallback provider.

    Search: DuckDuckGo HTML endpoint (no API key required).
    Fetch:  Direct HTTP request with content extraction.
    """

    def search(
        self,
        query: str,
        max_results: int,
        search_type: str,
        include_domains: list[str] | None,
    ) -> dict[str, Any]:
        del search_type  # DuckDuckGo has one search mode
        if not _HAS_BS4:
            raise ProviderError(
                "BeautifulSoup4 is required for the keyless fallback. "
                "Install with: pip install beautifulsoup4",
                "configuration",
            )

        try:
            response = _retry_transient(
                lambda: requests.post(
                    _DD_SEARCH_URL,
                    data={"q": query, "b": ""},
                    headers={
                        "User-Agent": _FETCH_USER_AGENT,
                        "Content-Type": "application/x-www-form-urlencoded",
                    },
                    timeout=_DD_TIMEOUT,
                    allow_redirects=True,
                ),
                max_retries=_DD_MAX_RETRIES,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"DuckDuckGo search request failed: {_redact_error(str(exc))}",
                "timeout" if isinstance(exc, requests.Timeout) else "api",
            ) from exc

        if response.status_code >= 400:
            raise ProviderError(
                f"DuckDuckGo search failed with status {response.status_code}",
                "api",
            )

        if "anomaly" in response.text.lower() and "captcha" in response.text.lower():
            raise ProviderError(
                "DuckDuckGo returned a CAPTCHA challenge. "
                "Try again later or set EXA_API_KEY for a more reliable provider.",
                "api",
            )

        results = self._parse_ddg_html(response.text, max_results, include_domains)
        return {"query": query, "results": results}

    def fetch(self, url: str, max_characters: int) -> dict[str, Any]:
        if not _HAS_BS4:
            raise ProviderError(
                "BeautifulSoup4 is required for the keyless fallback. "
                "Install with: pip install beautifulsoup4",
                "configuration",
            )

        normalized_url = _normalize_url(url)

        try:
            response = _retry_transient(
                lambda: requests.get(
                    normalized_url,
                    headers={
                        "User-Agent": _FETCH_USER_AGENT,
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7",
                        "Accept-Language": "en-US,en;q=0.9",
                    },
                    timeout=_FETCH_TIMEOUT,
                    allow_redirects=True,
                ),
                max_retries=_FETCH_MAX_RETRIES,
            )
        except requests.RequestException as exc:
            raise ProviderError(
                f"Fetch request failed: {_redact_error(str(exc))}",
                "timeout" if isinstance(exc, requests.Timeout) else "api",
            ) from exc

        final_url = response.url
        try:
            _validate_url(final_url)
        except UrlValidationError:
            raise ProviderError(
                f"Redirected to blocked URL: {final_url}", "validation"
            ) from None

        if response.status_code >= 400:
            raise ProviderError(
                f"Fetch failed with status {response.status_code} for {final_url}",
                "api",
            )

        content_type = response.headers.get("Content-Type", "").lower()

        if "text/html" not in content_type and "application/xhtml" not in content_type:
            raw = response.text or ""
            raw = raw[:max_characters]
            return {
                "url": final_url,
                "title": None,
                "markdown_content": raw,
                "content_length": len(raw),
            }

        html = response.text

        if _HAS_TRAFILATURA:
            extracted = trafilatura.extract(
                html,
                include_links=True,
                include_tables=True,
                favor_recall=True,
                url=final_url,
            )
            if extracted and extracted.strip():
                text = extracted[:max_characters]
                title = self._extract_title_bs4(html)
                return {
                    "url": final_url,
                    "title": title,
                    "markdown_content": text,
                    "content_length": len(text),
                }

        text, title = self._extract_content_bs4(html, max_characters)
        if not text.strip():
            raise ProviderError(
                "Could not extract readable content from the page", "api"
            )

        return {
            "url": final_url,
            "title": title,
            "markdown_content": text,
            "content_length": len(text),
        }

    @staticmethod
    def _parse_ddg_html(
        html: str,
        max_results: int,
        include_domains: list[str] | None,
    ) -> list[dict[str, Any]]:
        soup = BeautifulSoup(html, "html.parser")
        results: list[dict[str, Any]] = []

        for result_div in soup.select(".result"):
            link = result_div.select_one(".result__a")
            if not link:
                continue

            href = link.get("href", "")
            href = DuckDuckGoProvider._resolve_ddg_url(href)
            if not href:
                continue

            if include_domains:
                hostname = (urlparse(href).hostname or "").lower()
                if not any(_domain_matches(hostname, d) for d in include_domains):
                    continue

            title = link.get_text(strip=True)

            snippet_elem = result_div.select_one(".result__snippet")
            snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""

            results.append(
                {
                    "title": title,
                    "url": href,
                    "content": snippet,
                    "published_date": None,
                }
            )

            if len(results) >= max_results:
                break

        return results

    @staticmethod
    def _resolve_ddg_url(href: str) -> str:
        """Resolve DuckDuckGo redirect URLs to the actual target."""
        if not href:
            return ""

        if "uddg=" in href:
            parsed = urlparse(href)
            params = parse_qs(parsed.query)
            if "uddg" in params and params["uddg"]:
                return unquote(params["uddg"][0])

        if href.startswith("http://") or href.startswith("https://"):
            return href

        return ""

    @staticmethod
    def _extract_title_bs4(html: str) -> str | None:
        soup = BeautifulSoup(html, "html.parser")
        if soup.title and soup.title.string:
            return soup.title.string.strip()
        h1 = soup.find("h1")
        if h1:
            return h1.get_text(strip=True)
        return None

    @staticmethod
    def _extract_content_bs4(html: str, max_characters: int) -> tuple[str, str | None]:
        soup = BeautifulSoup(html, "html.parser")

        title = None
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        for element in soup(
            ["script", "style", "noscript", "nav", "footer", "header",
             "aside", "form", "iframe", "svg", "canvas"]
        ):
            element.decompose()

        for pattern in ["ad", "ads", "advertisement", "sidebar", "comment",
                        "comments", "share", "social", "newsletter", "cookie",
                        "popup", "modal", "banner", "skip"]:
            for elem in soup.find_all(attrs={"class": re.compile(pattern, re.I)}):
                elem.decompose()
            for elem in soup.find_all(attrs={"id": re.compile(pattern, re.I)}):
                elem.decompose()

        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find(attrs={"role": "main"})
            or soup.find("div", class_=re.compile(r"(content|main|post|article|entry)", re.I))
            or soup.body
            or soup
        )

        text = main.get_text(separator="\n", strip=True)

        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        if len(text) > max_characters:
            text = text[:max_characters]

        return text, title


_provider: SearchProvider | None = None


def reset_provider() -> None:
    """Clear the cached provider (e.g. after EXA_API_KEY changes)."""
    global _provider
    _provider = None


def get_provider() -> SearchProvider:
    """Return the active search provider, creating it on first call."""
    global _provider
    if _provider is None:
        if _has_exa():
            _provider = ExaProvider()
        else:
            _provider = DuckDuckGoProvider()
    return _provider


def _provider_name() -> str:
    return "exa" if _has_exa() else "duckduckgo"


@tool
def web_search(
    query: str,
    max_results: int = 5,
    search_type: Literal["auto", "instant", "deep-lite", "deep", "deep-reasoning"] = "auto",
    include_domains: list[str] | None = None,
) -> str:
    """Search the web for current information and documentation.

    After receiving results, synthesize the information into a helpful response and cite sources.
    Use include_domains to scope searches (e.g. ["docs.python.org", "github.com"]).

    Uses Exa when EXA_API_KEY is configured, otherwise falls back to DuckDuckGo (keyless).
    Note: search_type and include_domains are best-effort on the fallback provider.
    """
    validation_error = _validate_search_args(query, max_results)
    if validation_error:
        return _json_response(
            {"error": validation_error, "query": query, "category": "validation"}
        )

    provider = get_provider()
    try:
        result = provider.search(query, max_results, search_type, include_domains)
        result["provider"] = _provider_name()
        return _json_response(result)
    except ProviderError as exc:
        if exc.category == "configuration":
            return _json_response(
                {
                    "error": str(exc),
                    "query": query,
                    "category": "configuration",
                    "provider": _provider_name(),
                }
            )
        return _json_response(
            {
                "error": f"Web search error: {exc}",
                "query": query,
                "category": exc.category,
                "provider": _provider_name(),
            }
        )


@tool
def fetch_url(url: str, max_characters: int = 12000) -> str:
    """Fetch content from a URL.

    Use after web_search when you need full page content beyond search snippets.
    After receiving content, synthesize relevant information for the user.

    Uses Exa when EXA_API_KEY is configured, otherwise falls back to direct HTTP fetch.
    Note: the keyless fallback cannot render JavaScript-heavy pages.
    """
    validation_error = _validate_fetch_args(max_characters)
    if validation_error:
        return _json_response(
            {"error": validation_error, "url": url, "category": "validation"}
        )

    try:
        normalized_url = _normalize_url(url)
    except UrlValidationError as exc:
        return _json_response(
            {"error": str(exc), "url": url, "category": "validation"}
        )

    provider = get_provider()
    try:
        result = provider.fetch(normalized_url, max_characters)
        result["provider"] = _provider_name()
        return _json_response(result)
    except ProviderError as exc:
        return _json_response(
            {
                "error": f"Fetch URL error: {exc}",
                "url": normalized_url,
                "category": exc.category,
                "provider": _provider_name(),
            }
        )
