"""Middleware behavior: correlation id propagation + access log.

Offline; uses FastAPI TestClient. Doesn't touch Agent/RAG.
"""
import logging

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.middleware import REQUEST_ID_HEADER, CorrelationIdMiddleware
from app.core.logging import get_correlation_id

_EXTRA_KEYS = ("cid", "method", "path", "status", "dur_ms", "client")


def _app(quiet_paths_logged: bool = False) -> FastAPI:
    app = FastAPI()
    app.add_middleware(
        CorrelationIdMiddleware, log_quiet_paths=quiet_paths_logged
    )

    @app.get("/echo-cid")
    def echo_cid():
        return {"cid": get_correlation_id()}

    @app.get("/boom")
    def boom():
        raise RuntimeError("kaboom")

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    return app


def _records(caplog) -> list[dict]:
    """Extract structured records: message + extra fields."""
    out = []
    for r in caplog.records:
        if r.name != "app.api.access":
            continue
        rec = {"msg": r.getMessage(), "level": r.levelname}
        for k in _EXTRA_KEYS:
            if hasattr(r, k):
                rec[k] = getattr(r, k)
        out.append(rec)
    return out


# ---------- correlation id propagation ----------

def test_generates_cid_when_absent():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/echo-cid")
    assert r.status_code == 200
    cid = r.json()["cid"]
    assert cid and len(cid) <= 64
    assert r.headers[REQUEST_ID_HEADER] == cid


def test_uses_incoming_cid():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/echo-cid", headers={REQUEST_ID_HEADER: "req-abc-123"})
    assert r.json()["cid"] == "req-abc-123"
    assert r.headers[REQUEST_ID_HEADER] == "req-abc-123"


def test_rejects_weird_cid_and_generates():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/echo-cid", headers={REQUEST_ID_HEADER: "bad cid with spaces!"})
    cid = r.json()["cid"]
    assert cid != "bad cid with spaces!"
    assert cid


def test_rejects_overlong_cid():
    c = TestClient(_app(), raise_server_exceptions=False)
    too_long = "a" * 200
    r = c.get("/echo-cid", headers={REQUEST_ID_HEADER: too_long})
    assert r.json()["cid"] != too_long
    assert len(r.json()["cid"]) <= 64


# ---------- access logging ----------

def test_logs_ok_request(caplog):
    caplog.set_level(logging.INFO, logger="app.api.access")
    c = TestClient(_app(), raise_server_exceptions=False)
    c.get("/echo-cid")
    recs = _records(caplog)
    ok = [r for r in recs if r["msg"] == "request.ok"]
    assert len(ok) == 1
    assert ok[0]["method"] == "GET"
    assert ok[0]["path"] == "/echo-cid"
    assert ok[0]["status"] == 200
    assert "dur_ms" in ok[0]
    assert ok[0]["cid"]


def test_quiet_paths_not_logged_by_default(caplog):
    caplog.set_level(logging.INFO, logger="app.api.access")
    c = TestClient(_app(), raise_server_exceptions=False)
    c.get("/healthz")
    recs = _records(caplog)
    assert all(r.get("path") != "/healthz" for r in recs)


def test_quiet_paths_logged_when_enabled(caplog):
    caplog.set_level(logging.INFO, logger="app.api.access")
    c = TestClient(_app(quiet_paths_logged=True), raise_server_exceptions=False)
    c.get("/healthz")
    recs = _records(caplog)
    assert any(r.get("path") == "/healthz" for r in recs)


# ---------- error path ----------

def test_error_still_returns_cid_header():
    c = TestClient(_app(), raise_server_exceptions=False)
    r = c.get("/boom")
    assert r.status_code == 500
    assert r.headers[REQUEST_ID_HEADER]
    body = r.json()
    assert body["error"] == "internal_error"
    assert body["correlation_id"] == r.headers[REQUEST_ID_HEADER]


def test_error_logged(caplog):
    caplog.set_level(logging.ERROR, logger="app.api.access")
    c = TestClient(_app(), raise_server_exceptions=False)
    c.get("/boom")
    recs = _records(caplog)
    err = [r for r in recs if r["msg"] == "request.error"]
    assert len(err) == 1
    assert err[0]["path"] == "/boom"
    assert "dur_ms" in err[0]
    assert err[0]["cid"]
