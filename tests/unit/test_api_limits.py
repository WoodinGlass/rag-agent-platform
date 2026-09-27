"""Request limits: token bucket, body size, and their middleware wrappers.

Offline and deterministic: a fake clock drives the bucket, and test apps
set `app.state.settings` directly instead of relying on env vars.
"""
from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.limits import (
    TokenBucketLimiter,
    check_content_length,
)
from app.api.middleware import (
    BodySizeLimitMiddleware,
    CorrelationIdMiddleware,
    RateLimitMiddleware,
)
from app.core.config import Settings

# ---------- a fake clock ----------

class _Clock:
    def __init__(self, t0: float = 0.0) -> None:
        self.t = t0

    def now(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


# ---------- TokenBucketLimiter ----------

def test_bucket_allows_up_to_burst_then_denies():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=3, now_fn=clk.now)
    for _ in range(3):
        ok, retry = lim.allow("k")
        assert ok is True
        assert retry == 0.0
    ok, retry = lim.allow("k")
    assert ok is False
    assert retry > 0


def test_bucket_refills_over_time():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=2.0, burst=2, now_fn=clk.now)
    assert lim.allow("k")[0] is True
    assert lim.allow("k")[0] is True
    assert lim.allow("k")[0] is False
    clk.advance(0.5)  # 0.5s * 2 rps = 1 token
    assert lim.allow("k")[0] is True
    assert lim.allow("k")[0] is False


def test_bucket_refill_capped_at_burst():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=100.0, burst=2, now_fn=clk.now)
    clk.advance(60.0)  # 6000 tokens worth, capped to 2
    assert lim.allow("k")[0] is True
    assert lim.allow("k")[0] is True
    assert lim.allow("k")[0] is False


def test_bucket_keys_are_isolated():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=1, now_fn=clk.now)
    assert lim.allow("a")[0] is True
    assert lim.allow("a")[0] is False
    # different key has its own bucket
    assert lim.allow("b")[0] is True


def test_retry_after_matches_deficit():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=2.0, burst=1, now_fn=clk.now)
    lim.allow("k")  # consumes the single token
    ok, retry = lim.allow("k")
    assert ok is False
    # deficit is 1 token at 2 rps -> 0.5s
    assert 0.49 < retry < 0.51


def test_prune_drops_full_buckets_only():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=2, now_fn=clk.now)
    lim.allow("full")    # 1/2
    lim.allow("full")    # 0/2
    lim.allow("drained") # 1/2, drained
    clk.advance(10.0)    # both refill to burst
    removed = lim.prune()
    assert removed == 2
    assert lim.size == 0


def test_prune_keeps_still_depleted_buckets():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=3, now_fn=clk.now)
    for _ in range(3):
        lim.allow("k")  # 0/3
    clk.advance(0.5)    # 0.5/3 -> still not full
    assert lim.prune() == 0
    assert lim.size == 1


def test_bucket_validation():
    with pytest.raises(ValueError, match="rate must be > 0"):
        TokenBucketLimiter(rate=0, burst=1)
    with pytest.raises(ValueError, match="burst must be >= 1"):
        TokenBucketLimiter(rate=1, burst=0)


# ---------- check_content_length ----------

def test_content_length_disabled_when_max_zero():
    v = check_content_length("999999", 0)
    assert v.allowed is True


def test_content_length_missing_allowed():
    v = check_content_length(None, 1024)
    assert v.allowed is True


def test_content_length_within_limit():
    v = check_content_length("100", 1024)
    assert v.allowed is True
    assert v.declared == 100


def test_content_length_exceeds_limit():
    v = check_content_length("2048", 1024)
    assert v.allowed is False
    assert v.declared == 2048
    assert v.field["max_bytes"] == 1024
    assert v.field["declared_bytes"] == 2048


def test_content_length_invalid_rejected():
    v = check_content_length("not-a-number", 1024)
    assert v.allowed is False
    assert "invalid" in v.reason


def test_content_length_negative_rejected():
    v = check_content_length("-1", 1024)
    assert v.allowed is False
    assert "negative" in v.reason


# ---------- middleware helpers ----------

def _app(*, rate_enabled=False, rps=1.0, burst=1, max_body=0, limiter=None):
    settings = Settings(
        rate_limit_enabled=rate_enabled,
        rate_limit_rps=rps,
        rate_limit_burst=burst,
        max_body_size_bytes=max_body,
    )
    app = FastAPI()
    app.state.settings = settings
    app.state.rate_limiter = limiter
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(CorrelationIdMiddleware)

    @app.post("/probe")
    async def probe():
        return {"ok": True}

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    return app


# ---------- rate limit middleware ----------

def test_rate_limit_disabled_by_default():
    c = TestClient(_app(rate_enabled=False))
    for _ in range(5):
        assert c.post("/probe").status_code == 200


def test_rate_limit_429_after_burst():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=2, now_fn=clk.now)
    c = TestClient(_app(rate_enabled=True, limiter=lim))

    assert c.post("/probe").status_code == 200
    assert c.post("/probe").status_code == 200
    r = c.post("/probe")
    assert r.status_code == 429
    assert "Retry-After" in r.headers
    assert int(r.headers["Retry-After"]) >= 1
    body = r.json()
    assert body["error"] == "rate_limited"
    assert body["retry_after_s"] >= 1


def test_rate_limit_isolates_by_api_key():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=1, now_fn=clk.now)
    c = TestClient(_app(rate_enabled=True, limiter=lim))

    assert c.post("/probe", headers={"X-API-Key": "a"}).status_code == 200
    assert c.post("/probe", headers={"X-API-Key": "a"}).status_code == 429
    # different key -> its own bucket
    assert c.post("/probe", headers={"X-API-Key": "b"}).status_code == 200


def test_rate_limit_exempts_healthz():
    clk = _Clock()
    lim = TokenBucketLimiter(rate=1.0, burst=1, now_fn=clk.now)
    c = TestClient(_app(rate_enabled=True, limiter=lim))

    # consume the bucket
    assert c.post("/probe").status_code == 200
    assert c.post("/probe").status_code == 429
    # healthz still reachable
    for _ in range(5):
        assert c.get("/healthz").status_code == 200


# ---------- body size middleware ----------

def test_body_size_disabled_by_default():
    c = TestClient(_app(max_body=0))
    r = c.post("/probe", content=b"x" * 10_000)
    assert r.status_code == 200


def test_body_size_413_when_over_limit():
    c = TestClient(_app(max_body=100))
    r = c.post("/probe", content=b"x" * 200)
    assert r.status_code == 413
    body = r.json()
    assert body["error"] == "payload_too_large"
    assert body["max_bytes"] == 100
    assert body["declared_bytes"] == 200


def test_body_size_under_limit_ok():
    c = TestClient(_app(max_body=100))
    assert c.post("/probe", content=b"x" * 50).status_code == 200


def test_body_size_413_carries_correlation_id():
    c = TestClient(_app(max_body=10))
    r = c.post(
        "/probe",
        content=b"x" * 100,
        headers={"X-Request-ID": "trace-1"},
    )
    assert r.status_code == 413
    assert r.json()["correlation_id"] == "trace-1"
