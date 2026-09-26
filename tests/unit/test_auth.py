"""Tenant auth: parse, resolve, and error paths. Offline."""
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.auth import AuthError, parse_tenant_keys, resolve_tenant
from app.core.config import Settings

# ---------- parse_tenant_keys ----------

def test_parse_empty():
    assert parse_tenant_keys("") == {}


def test_parse_single():
    assert parse_tenant_keys("k1:acme") == {"k1": "acme"}


def test_parse_multiple():
    assert parse_tenant_keys("k1:acme,k2:globex") == {"k1": "acme", "k2": "globex"}


def test_parse_whitespace():
    assert parse_tenant_keys("  k1 : acme , k2:globex  ") == {
        "k1": "acme",
        "k2": "globex",
    }


def test_parse_skips_empty_entries():
    assert parse_tenant_keys("k1:acme,,k2:globex") == {"k1": "acme", "k2": "globex"}


def test_parse_rejects_missing_colon():
    with pytest.raises(AuthError, match="invalid tenant key entry"):
        parse_tenant_keys("k1acme")


def test_parse_rejects_empty_key():
    with pytest.raises(AuthError, match="invalid tenant key entry"):
        parse_tenant_keys(":acme")


def test_parse_rejects_empty_tenant():
    with pytest.raises(AuthError, match="invalid tenant key entry"):
        parse_tenant_keys("k1:")


def test_parse_rejects_duplicate():
    with pytest.raises(AuthError, match="duplicate tenant key"):
        parse_tenant_keys("k1:acme,k1:globex")


# ---------- resolve_tenant ----------

def _settings(**over) -> Settings:
    base = {
        "tenant_auth_enabled": False,
        "tenant_keys": "",
        "default_tenant": "public",
    }
    base.update(over)
    return Settings(**base)


def test_resolve_auth_disabled_returns_default():
    s = _settings(tenant_auth_enabled=False, default_tenant="public")
    assert resolve_tenant(None, s) == "public"
    assert resolve_tenant("anything", s) == "public"


def test_resolve_auth_enabled_valid_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme,k2:globex")
    assert resolve_tenant("k1", s) == "acme"
    assert resolve_tenant("k2", s) == "globex"


def test_resolve_auth_enabled_missing_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme")
    with pytest.raises(AuthError, match="missing X-API-Key"):
        resolve_tenant(None, s)


def test_resolve_auth_enabled_invalid_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme")
    with pytest.raises(AuthError, match="invalid X-API-Key"):
        resolve_tenant("nope", s)


def test_resolve_auth_enabled_no_keys_fails_closed():
    s = _settings(tenant_auth_enabled=True, tenant_keys="")
    with pytest.raises(AuthError, match="no tenant keys configured"):
        resolve_tenant("k1", s)


# ---------- HTTP integration (route dep) ----------

def _client(settings: Settings) -> TestClient:
    from app.agents.bootstrap import build_agent
    from app.agents.llm import FakeLLM
    from app.api.deps import get_app_settings
    from app.api.middleware import CorrelationIdMiddleware, TenantMiddleware
    from app.api.routes import router
    from app.rag.embedder import FakeEmbedder
    from app.rag.store import MemoryStore

    app = FastAPI()
    app.state.settings = settings  # visible to TenantMiddleware
    app.add_middleware(TenantMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_app_settings] = lambda: settings

    stack = build_agent(
        llm=FakeLLM(['{"action":"finish","final_answer":"ok"}']),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )
    app.state.agent = stack.agent
    app.state.pipeline = stack.pipeline
    return TestClient(app, raise_server_exceptions=False)


def test_ingest_401_when_auth_on_and_no_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme")
    c = _client(s)
    r = c.post(
        "/ingest",
        json={"content_base64": "aGVsbG8=", "source": "x.txt"},
    )
    assert r.status_code == 401
    assert "X-API-Key" in r.json()["detail"]


def test_ingest_401_when_auth_on_and_bad_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme")
    c = _client(s)
    r = c.post(
        "/ingest",
        headers={"X-API-Key": "wrong"},
        json={"content_base64": "aGVsbG8=", "source": "x.txt"},
    )
    assert r.status_code == 401


def test_ingest_200_when_auth_on_and_good_key():
    s = _settings(tenant_auth_enabled=True, tenant_keys="k1:acme")
    c = _client(s)
    r = c.post(
        "/ingest",
        headers={"X-API-Key": "k1"},
        json={"content_base64": "aGVsbG8=", "source": "x.txt"},
    )
    assert r.status_code == 200
    assert r.json()["doc_id"]


def test_ingest_200_when_auth_off():
    s = _settings(tenant_auth_enabled=False)
    c = _client(s)
    r = c.post("/ingest", json={"content_base64": "aGVsbG8="})
    assert r.status_code == 200


def test_healthz_reports_tenant_auth_flag():
    c = _client(_settings(tenant_auth_enabled=True, tenant_keys="k1:acme"))
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["checks"]["tenant_auth"] is True


def test_healthz_reports_tenant_auth_flag_off():
    c = _client(_settings(tenant_auth_enabled=False))
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["checks"]["tenant_auth"] is False
