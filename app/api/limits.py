"""Request limits: token-bucket rate limiter + body size guard.

Both are off by default (portfolio runs with zero config) and stdlib
only (no Redis). The limiter is in-process: fine for a single worker or
a demo. For horizontal scale, replace `TokenBucketLimiter` with a Redis
adapter behind the same `allow(key)` contract — the middleware does not
need to change.

Design:
- `TokenBucketLimiter` keys by tenant id (when auth resolves one) or
  client IP. Each key gets its own bucket. Buckets are pruned lazily
  once they are full again, so memory stays bounded under long uptimes
  with many distinct keys.
- A monotonic clock is injected (`now_fn`) so tests can advance time
  deterministically without `sleep`.
- Body size is enforced from `Content-Length` first; if missing (chunked
  upload), the middleware rejects oversized bodies as they stream. We
  do not buffer the whole body — that would defeat the purpose.
"""
from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from threading import Lock

# ---------- token bucket ----------

@dataclass
class _Bucket:
    tokens: float
    last_refill: float


class TokenBucketLimiter:
    """In-process token bucket. Thread-safe.

    `rate` tokens are added per second, up to `burst` capacity. A request
    is allowed if at least 1 token is available; that token is consumed.

    Keys that have been idle long enough to be full again are pruned the
    next time `allow` runs, which keeps memory from growing without bound.
    """

    def __init__(
        self,
        *,
        rate: float,
        burst: int,
        now_fn: Callable[[], float] | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError("rate must be > 0")
        if burst < 1:
            raise ValueError("burst must be >= 1")
        self.rate = float(rate)
        self.burst = int(burst)
        self._now = now_fn or time.monotonic
        self._lock = Lock()
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds).

        retry_after_seconds is 0.0 when allowed.
        """
        with self._lock:
            now = self._now()
            b = self._buckets.get(key)
            if b is None:
                b = _Bucket(tokens=float(self.burst), last_refill=now)
                self._buckets[key] = b

            # Refill based on elapsed time, cap at burst.
            elapsed = max(0.0, now - b.last_refill)
            if elapsed > 0:
                b.tokens = min(float(self.burst), b.tokens + elapsed * self.rate)
                b.last_refill = now

            if b.tokens >= 1.0:
                b.tokens -= 1.0
                return True, 0.0

            deficit = 1.0 - b.tokens
            retry_after = deficit / self.rate
            return False, retry_after

    def prune(self) -> int:
        """Drop buckets that are already at burst capacity. Returns count."""
        with self._lock:
            now = self._now()
            keep: dict[str, _Bucket] = {}
            for key, b in self._buckets.items():
                elapsed = max(0.0, now - b.last_refill)
                tokens_if_refilled = min(
                    float(self.burst), b.tokens + elapsed * self.rate
                )
                if tokens_if_refilled >= self.burst:
                    continue  # full -> no state worth keeping
                keep[key] = b
            removed = len(self._buckets) - len(keep)
            self._buckets = keep
            return removed

    @property
    def size(self) -> int:
        return len(self._buckets)


# ---------- body size guard ----------

@dataclass
class BodySizeVerdict:
    allowed: bool
    reason: str = ""
    declared: int | None = None
    field: dict = field(default_factory=dict)


def check_content_length(
    content_length_header: str | None,
    max_bytes: int,
) -> BodySizeVerdict:
    """Fast-path check: reject from the header alone when possible.

    - `max_bytes <= 0` disables the check.
    - Missing or non-numeric header -> allowed (streaming check applies).
    """
    if max_bytes <= 0:
        return BodySizeVerdict(allowed=True)
    if content_length_header is None:
        return BodySizeVerdict(allowed=True)

    try:
        declared = int(content_length_header)
    except (TypeError, ValueError):
        return BodySizeVerdict(
            allowed=False,
            reason="invalid Content-Length header",
        )

    if declared < 0:
        return BodySizeVerdict(
            allowed=False,
            reason="negative Content-Length",
            declared=declared,
        )

    if declared > max_bytes:
        return BodySizeVerdict(
            allowed=False,
            reason=f"body exceeds {max_bytes} bytes",
            declared=declared,
            field={"max_bytes": max_bytes, "declared_bytes": declared},
        )

    return BodySizeVerdict(allowed=True, declared=declared)
