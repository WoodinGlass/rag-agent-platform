"""Tenant contextvar: set/get/reset + downstream propagation.

Offline. Verifies that the adapter reads the tenant from context and
that the HTTP layer sets it per-request.
"""
import base64

from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.api.middleware import CorrelationIdMiddleware, TenantMiddleware
from app.api.routes import router
from app.core.config import Settings
from app.core.tenant import get_tenant_id, reset_tenant_id, set_tenant_id
from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore
from app.tools.adapters.rag import make_retrieve_fn
from app.tools.base import Tool

# ---------- contextvar basics ----------

def test_default_none():
    assert get_tenant_id() is None


def test_set_and_get():
    token = set_tenant_id("acme")
    try:
        assert get_tenant_id() == "acme"
    finally:
        reset_tenant_id(token)
    assert get_tenant_id() is None


def test_reset_restores_previous():
    outer = set_tenant_id("outer")
    try:
        inner = set_tenant_id("inner")
        assert get_tenant_id() == "inner"
        reset_tenant_id(inner)
        assert get_tenant_id() == "outer"
    finally:
        reset_tenant_id(outer)


# ---------- adapter reads context ----------

def _pipe_with_tenants():
    pipe = RagPipeline(FakeEmbedder(dim=64), MemoryStore(), chunk_size=200)
    pipe.ingest(b"tenant A secret content", tenant_id="A")
    pipe.ingest(b"tenant B public content", tenant_id="B")
    return pipe


def test_adapter_uses_tenant_from_context():
    pipe = _pipe_with_tenants()
    retrieve = make_retrieve_fn(pipe)

    token = set_tenant_id("A")
    try:
        hits_a = retrieve("content", 10)
    finally:
        reset_tenant_id(token)

    token = set_tenant_id("B")
    try:
        hits_b = retrieve("content", 10)
    finally:
        reset_tenant_id(token)

    assert all(h.metadata.get("tenant_id") == "A" for h in hits_a)
    assert all(h.metadata.get("tenant_id") == "B" for h in hits_b)


def test_adapter_no_context_returns_all():
    pipe = _pipe_with_tenants()
    retrieve = make_retrieve_fn(pipe)
    hits = retrieve("content", 10)
    tenants = {h.metadata.get("tenant_id") for h in hits}
    assert tenants == {"A", "B"}


# ---------- HTTP: request tenant flows to search_docs ----------

def _settings_auth_on() -> Settings:
    return Settings(
        tenant_auth_enabled=True,
        tenant_keys="kA:A,kB:B",
        default_tenant="public",
    )


def _client_and_spy(settings: Settings):
    """Build a TestClient whose search_docs tool records the tenant it sees.

    The spy uses the SAME pipeline the routes use (`stack.pipeline`), so
    HTTP ingest and HTTP query share one store. A separate FakeLLM script
    is prepared per request.
    """
    # 6 LLM replies are enough for 2 requests, each doing tool -> finish.
    script = [
        '{"action":"tool","tool":"search_docs","args":{"query":"content"}}',
        '{"action":"finish","final_answer":"ok"}',
        '{"action":"tool","tool":"search_docs","args":{"query":"content"}}',
        '{"action":"finish","final_answer":"ok"}',
        '{"action":"tool","tool":"search_docs","args":{"query":"content"}}',
        '{"action":"finish","final_answer":"ok"}',
    ]
    stack = build_agent(
        llm=FakeLLM(list(script)),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )
    pipe = stack.pipeline  # routes and spy share this store

    seen: list[dict] = []

    def spy_search_docs(query: str, k: int = 5) -> dict:
        tenant = get_tenant_id()
        seen.append({"query": query, "tenant": tenant})
        hits = pipe.retrieve(query, k=k, tenant_id=tenant)
        return {"n": len(hits), "hits": [h.text for h in hits]}

    spy_tool = Tool(
        name="search_docs",
        description="spy",
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
        func=spy_search_docs,
    )
    stack.registry.unregister("search_docs")
    stack.registry.register(spy_tool)

    from app.api.deps import get_app_settings

    app = FastAPI()
    app.state.settings = settings  # visible to TenantMiddleware
    app.add_middleware(TenantMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    app.dependency_overrides[get_app_settings] = lambda: settings
    app.state.agent = stack.agent
    app.state.pipeline = stack.pipeline
    app.state.stack = stack

    return TestClient(app, raise_server_exceptions=False), seen, pipe


def test_http_query_exposes_request_tenant_to_tools():
    client, seen, _ = _client_and_spy(_settings_auth_on())

    r = client.post(
        "/query",
        headers={"X-API-Key": "kA"},
        json={"question": "what is content?"},
    )
    assert r.status_code == 200
    assert seen and seen[-1]["tenant"] == "A"

    r = client.post(
        "/query",
        headers={"X-API-Key": "kB"},
        json={"question": "what is content?"},
    )
    assert r.status_code == 200
    assert seen[-1]["tenant"] == "B"


def test_http_cross_tenant_no_leak_through_agent():
    client, _, pipe = _client_and_spy(_settings_auth_on())

    # tenant A ingests a secret
    payload = base64.b64encode(b"SECRET-PROJECT-XYZ confidential").decode()
    r = client.post(
        "/ingest",
        headers={"X-API-Key": "kA"},
        json={"content_base64": payload, "source": "s.txt"},
    )
    assert r.status_code == 200

    # tenant B queries for it; search_docs should not return A's chunk
    r = client.post(
        "/query",
        headers={"X-API-Key": "kB"},
        json={"question": "SECRET-PROJECT-XYZ"},
    )
    assert r.status_code == 200

    # Verify at store level (deterministic, independent of LLM output)
    hits_b = pipe.retrieve("SECRET", k=10, tenant_id="B")
    assert all("SECRET-PROJECT-XYZ" not in h.text for h in hits_b)

    hits_a = pipe.retrieve("SECRET", k=10, tenant_id="A")
    assert any("SECRET-PROJECT-XYZ" in h.text for h in hits_a)


def test_http_single_tenant_uses_default():
    settings = Settings(
        tenant_auth_enabled=False,
        default_tenant="public",
    )
    client, seen, _ = _client_and_spy(settings)
    r = client.post("/query", json={"question": "hi"})
    assert r.status_code == 200
    assert seen[-1]["tenant"] == "public"
