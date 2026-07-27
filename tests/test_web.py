from __future__ import annotations

import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("OPENAI_API_KEY", "test")

import liteharness.tools.web as web
from liteharness.tools.web import (
    DuckDuckGoProvider,
    ExaProvider,
    ProviderError,
    UrlValidationError,
    _validate_url,
    get_provider,
    reset_provider,
    web_search,
    fetch_url,
)

from tests.sdk_fixtures import SessionContextTestMixin, set_exa_key

_SAMPLE_DDG_HTML = """
<html><body>
<div class="result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fdocs.python.org%2F3%2Flibrary%2Fasyncio.html">Asyncio docs</a>
  <div class="result__snippet">asyncio.run(main())</div>
</div>
</body></html>
"""


def _addr(ip: str) -> tuple:
    family = socket.AF_INET6 if ":" in ip else socket.AF_INET
    return (family, socket.SOCK_STREAM, 6, "", (ip, 0))


def _response(status_code: int = 200, payload: dict | None = None, text: str = "") -> mock.Mock:
    response = mock.Mock()
    response.status_code = status_code
    response.text = text
    response.headers = {"Content-Type": "text/html"}
    response.url = "https://docs.python.org/3/"
    response.json.return_value = payload if payload is not None else {}
    return response


class ValidateUrlTests(unittest.TestCase):
    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("file:///etc/passwd")

    def test_rejects_missing_hostname(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("https:///path")

    def test_rejects_localhost(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("http://localhost/docs")

    def test_rejects_loopback_ip(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("http://127.0.0.1/")

    def test_rejects_metadata_ip(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("http://169.254.169.254/latest/meta-data")

    def test_rejects_userinfo(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("https://user:pass@example.com/")

    def test_rejects_dns_private_resolution(self) -> None:
        with mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("10.0.0.5")]):
            with self.assertRaises(UrlValidationError):
                _validate_url("https://169.254.169.254.nip.io/")

    def test_rejects_dns_loopback_resolution_for_odd_hostname(self) -> None:
        with mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("127.0.0.1")]):
            with self.assertRaises(UrlValidationError):
                _validate_url("http://2130706433/")

    def test_rejects_private_ipv6_literal(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("http://[fc00::1]/")

    def test_rejects_invalid_port(self) -> None:
        with self.assertRaises(UrlValidationError):
            _validate_url("https://example.com:bad/")

    def test_accepts_public_url(self) -> None:
        with mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("93.184.216.34")]):
            _validate_url("https://docs.python.org/3/")


class WebSearchTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name), exa_api_key=None)
        reset_provider()

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_fallback_without_exa_key(self) -> None:
        with mock.patch.object(web.requests, "post", return_value=_response(200, text=_SAMPLE_DDG_HTML)):
            result = json.loads(web_search.invoke({"query": "python asyncio"}))

        self.assertEqual(result["provider"], "duckduckgo")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["title"], "Asyncio docs")
        self.assertIn("asyncio.run", result["results"][0]["content"])

    def test_rejects_empty_query(self) -> None:
        result = json.loads(web_search.invoke({"query": "   "}))
        self.assertEqual(result["category"], "validation")
        self.assertIn("query", result["error"])

    def test_rejects_out_of_range_max_results(self) -> None:
        result = json.loads(web_search.invoke({"query": "python", "max_results": 11}))
        self.assertEqual(result["category"], "validation")
        self.assertIn("max_results", result["error"])

    def test_success_normalizes_results(self) -> None:
        response = {
            "results": [
                {
                    "title": "Asyncio docs",
                    "url": "https://docs.python.org/3/library/asyncio.html",
                    "highlights": ["asyncio.run(main())"],
                    "publishedDate": "2024-01-01",
                }
            ]
        }
        set_exa_key(self.ctx, "test")
        reset_provider()
        with mock.patch.object(ExaProvider, "_request", return_value=response) as exa_request:
            result = json.loads(web_search.invoke({"query": "python asyncio"}))

        self.assertEqual(result["provider"], "exa")
        self.assertEqual(result["query"], "python asyncio")
        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["title"], "Asyncio docs")
        self.assertIn("asyncio.run", result["results"][0]["content"])
        exa_request.assert_called_once()
        self.assertEqual(exa_request.call_args.args[0], "/search")
        self.assertEqual(exa_request.call_args.args[1]["numResults"], 5)

    def test_api_error_is_redacted(self) -> None:
        set_exa_key(self.ctx, "test")
        reset_provider()
        with mock.patch.object(
            ExaProvider,
            "_request",
            side_effect=ProviderError("status 500: secret body"),
        ):
            result = json.loads(web_search.invoke({"query": "python"}))
        self.assertEqual(result["category"], "api")
        self.assertIn("Web search error", result["error"])


class FetchUrlTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name), exa_api_key=None)
        reset_provider()

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_validation_failure(self) -> None:
        result = json.loads(fetch_url.invoke({"url": "http://127.0.0.1/secret"}))
        self.assertEqual(result["category"], "validation")
        self.assertIn("error", result)

    def test_rejects_out_of_range_max_characters(self) -> None:
        result = json.loads(fetch_url.invoke({"url": "https://example.com", "max_characters": 0}))
        self.assertEqual(result["category"], "validation")
        self.assertIn("max_characters", result["error"])

    def test_fallback_fetch_without_exa_key(self) -> None:
        page_html = "<html><head><title>Python docs</title></head><body><main>Overview</main></body></html>"
        fetch_response = _response(200, text=page_html)
        fetch_response.url = "https://docs.python.org/3/"

        with (
            mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("93.184.216.34")]),
            mock.patch.object(web.requests, "get", return_value=fetch_response),
            mock.patch.object(web.trafilatura, "extract", return_value="# Python 3 documentation\n\nOverview"),
        ):
            result = json.loads(fetch_url.invoke({"url": "https://docs.python.org/3/"}))

        self.assertEqual(result["provider"], "duckduckgo")
        self.assertIn("Python 3 documentation", result["markdown_content"])
        self.assertGreater(result["content_length"], 0)

    def test_success_returns_markdown(self) -> None:
        response = {
            "results": [
                {
                    "url": "https://docs.python.org/3/",
                    "title": "Python docs",
                    "text": "# Python 3 documentation\n\nOverview",
                }
            ],
            "statuses": [],
        }
        set_exa_key(self.ctx, "test")
        reset_provider()
        with (
            mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("93.184.216.34")]),
            mock.patch.object(ExaProvider, "_request", return_value=response) as exa_request,
        ):
            result = json.loads(fetch_url.invoke({"url": "https://docs.python.org/3/"}))

        self.assertEqual(result["provider"], "exa")
        self.assertEqual(result["title"], "Python docs")
        self.assertIn("Python 3 documentation", result["markdown_content"])
        self.assertGreater(result["content_length"], 0)
        self.assertEqual(exa_request.call_args.args[0], "/contents")
        self.assertEqual(exa_request.call_args.args[1]["urls"], ["https://docs.python.org/3/"])
        self.assertEqual(exa_request.call_args.args[1]["text"]["maxCharacters"], 12000)

    def test_api_error_from_statuses(self) -> None:
        set_exa_key(self.ctx, "test")
        reset_provider()
        with (
            mock.patch.object(web.socket, "getaddrinfo", return_value=[_addr("93.184.216.34")]),
            mock.patch.object(
                ExaProvider,
                "_request",
                side_effect=ProviderError("Exa could not fetch https://example.com", "api"),
            ),
        ):
            result = json.loads(fetch_url.invoke({"url": "https://example.com"}))

        self.assertEqual(result["category"], "api")
        self.assertIn("error", result)

    def test_exa_request_retries_transient_status(self) -> None:
        set_exa_key(self.ctx, "test")
        reset_provider()
        with (
            mock.patch.object(web.time, "sleep") as sleep,
            mock.patch.object(
                web.requests,
                "post",
                side_effect=[
                    _response(503, {"error": "temporary"}),
                    _response(200, {"results": []}),
                ],
            ) as post,
        ):
            result = ExaProvider()._request("/search", {"query": "python"})

        self.assertEqual(result, {"results": []})
        self.assertEqual(post.call_count, 2)
        sleep.assert_called_once()

    def test_exa_request_timeout_category(self) -> None:
        set_exa_key(self.ctx, "test")
        reset_provider()
        with (
            mock.patch.object(web.time, "sleep"),
            mock.patch.object(web.requests, "post", side_effect=web.requests.Timeout("timed out")),
        ):
            with self.assertRaises(ProviderError) as ctx:
                ExaProvider()._request("/search", {"query": "python"})

        self.assertEqual(ctx.exception.category, "timeout")

    def test_response_error_detail_is_redacted_and_truncated(self) -> None:
        detail = ExaProvider._response_error_detail(_response(500, {"error": "x" * 500}))
        self.assertLessEqual(len(detail), 300)
        self.assertTrue(detail.endswith("..."))

    def test_error_redaction_removes_api_key(self) -> None:
        set_exa_key(self.ctx, "secret-key")
        detail = web._redact_error("request failed with secret-key")
        self.assertNotIn("secret-key", detail)
        self.assertIn("[REDACTED]", detail)


class ProviderTests(SessionContextTestMixin, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.install_ctx(Path(self._tmp.name), exa_api_key=None)
        reset_provider()

    def tearDown(self) -> None:
        self.uninstall_ctx()
        self._tmp.cleanup()

    def test_get_provider_uses_exa_when_key_set(self) -> None:
        set_exa_key(self.ctx, "test")
        reset_provider()
        self.assertIsInstance(get_provider(), ExaProvider)

    def test_get_provider_uses_duckduckgo_without_key(self) -> None:
        self.assertIsInstance(get_provider(), DuckDuckGoProvider)

    def test_reset_provider_switches_after_settings_change(self) -> None:
        self.assertIsInstance(get_provider(), DuckDuckGoProvider)

        set_exa_key(self.ctx, "test")
        self.assertIsInstance(get_provider(), DuckDuckGoProvider)
        reset_provider()
        self.assertIsInstance(get_provider(), ExaProvider)

    def test_resolve_ddg_redirect_url(self) -> None:
        href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fpath"
        self.assertEqual(
            DuckDuckGoProvider._resolve_ddg_url(href),
            "https://example.com/path",
        )


if __name__ == "__main__":
    unittest.main()
