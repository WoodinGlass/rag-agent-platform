"""web_fetch tool.

Pure logic: validate url, call injected fetch_fn, format response.
Impure boundary: fetch_fn is injected (see tools/adapters/http.py).

Output shape (stable contract):
    {
      "url": str,
      "status": int,
      "ok": bool,
      "elapsed_s": float,
      "content_type": str,
      "text": str,
      "truncated": bool
    }
"""
from __future__ import annotations

from collections.abc import Callable
from urllib.parse import urlparse

from app.tools.adapters.http import HttpResponse
from app.tools.base import Tool

MAX_BYTES = 200_000
DEFAULT_TEXT_LIMIT = 4000

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "url": {
            "type": "string",
            "description": "Absolute http(s) URL to fetch.",
        },
        "max_chars": {
            "type": "integer",
            "description": f"Max chars of body returned (1..{DEFAULT_TEXT_LIMIT}).",
            "minimum": 1,
            "maximum": DEFAULT_TEXT_LIMIT,
        },
    },
    "required": ["url"],
}


def _validate_url(url: str) -> str:
    if not isinstance(url, str) or not url.strip():
        raise ValueError("url must be a non-empty string")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"url scheme must be http/https, got {parsed.scheme!r}")
    if not parsed.netloc:
        raise ValueError("url must have a host")
    return url


def _clamp_chars(n: int) -> int:
    return max(1, min(int(n), DEFAULT_TEXT_LIMIT))


def format_response(
    resp: HttpResponse, max_chars: int
) -> dict:
    """Pure: HttpResponse -> JSON-safe dict. No IO."""
    limit = _clamp_chars(max_chars)
    text = resp.text
    truncated = len(text) > limit
    return {
        "url": resp.url,
        "status": resp.status,
        "ok": resp.ok(),
        "elapsed_s": round(resp.elapsed_s, 4),
        "content_type": resp.headers.get("content-type", ""),
        "text": text[:limit],
        "truncated": truncated,
    }


def make_web_fetch_tool(
    fetch_fn: Callable[[str, int], HttpResponse],
    *,
    name: str = "web_fetch",
) -> Tool:
    """Build a Tool from an injected fetch function.

    `fetch_fn(url, max_bytes) -> HttpResponse` — swap for a fake in tests.
    """

    def _fetch(url: str, max_chars: int = DEFAULT_TEXT_LIMIT) -> dict:
        clean = _validate_url(url)
        limit = _clamp_chars(max_chars)
        resp = fetch_fn(clean, MAX_BYTES)
        return format_response(resp, limit)

    return Tool(
        name=name,
        description=(
            "Fetch a URL and return its body as text. Use for reading "
            "a specific page when the answer isn't in the local corpus."
        ),
        input_schema=INPUT_SCHEMA,
        func=_fetch,
        tags=("io", "http"),
    )
