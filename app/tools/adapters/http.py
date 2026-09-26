"""HTTP adapter: sync GET with timeout + retry + jitter.

Impure boundary for tools that need the network. Tools that use this
receive a callable `fetch(url) -> HttpResponse` and never import httpx
themselves, so unit tests can inject a fake.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HttpResponse:
    url: str
    status: int
    text: str
    headers: dict[str, str]
    elapsed_s: float

    def ok(self) -> bool:
        return 200 <= self.status < 300

    def json(self) -> Any:
        import json as _json
        return _json.loads(self.text)


class HttpError(RuntimeError):
    def __init__(self, msg: str, *, status: int | None = None) -> None:
        super().__init__(msg)
        self.status = status


def _sleep_backoff(attempt: int, base: float, cap: float) -> None:
    delay = min(cap, base * (2 ** attempt))
    delay += random.uniform(0, delay * 0.1)  # jitter 10%
    time.sleep(delay)


def make_http_fetch(
    *,
    timeout_s: float = 10.0,
    max_retries: int = 3,
    backoff_base_s: float = 0.25,
    backoff_cap_s: float = 4.0,
    user_agent: str = "rag-agent-platform/0.1",
    retry_statuses: tuple[int, ...] = (429, 500, 502, 503, 504),
):
    """Return a `fetch(url, max_bytes) -> HttpResponse` callable.

    Retries on: network errors, timeout, retry_statuses.
    Does not retry on 4xx (except 429).
    Raises `HttpError` after exhausting retries.
    """

    def fetch(url: str, max_bytes: int = 200_000) -> HttpResponse:
        import httpx

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            t0 = time.monotonic()
            try:
                r = httpx.get(
                    url,
                    timeout=timeout_s,
                    headers={"User-Agent": user_agent},
                    follow_redirects=True,
                )
                elapsed = time.monotonic() - t0
                if r.status_code in retry_statuses and attempt < max_retries:
                    _sleep_backoff(attempt, backoff_base_s, backoff_cap_s)
                    continue
                text = r.text
                if len(text.encode("utf-8")) > max_bytes:
                    text = text[:max_bytes]
                return HttpResponse(
                    url=str(r.url),
                    status=r.status_code,
                    text=text,
                    headers={k.lower(): v for k, v in r.headers.items()},
                    elapsed_s=elapsed,
                )
            except (httpx.TimeoutException, httpx.TransportError) as e:
                last_exc = e
                if attempt < max_retries:
                    _sleep_backoff(attempt, backoff_base_s, backoff_cap_s)
                    continue
                raise HttpError(f"network error after {max_retries + 1} tries: {e}") from e

        raise HttpError(f"unreachable: {last_exc}")

    return fetch
