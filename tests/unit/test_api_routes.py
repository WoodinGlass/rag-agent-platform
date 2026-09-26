"""Route-level behavior. Uses TestClient with app.state pre-populated.

Offline: FakeLLM + FakeEmbedder + MemoryStore.
"""
import base64

from fastapi.testclient import TestClient

from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.api.middleware import REQUEST_ID_HEADER
from app.main import create_app
from app.rag.embedder import FakeEmbedder
from app.rag.store import MemoryStore


def _client(script: list[str] | None = None) -> TestClient:
    script = script or ['{"action":"finish","final_answer":"ok","confidence":"high"}']
    app = create_app()

    # bypass lifespan: inject stack directly
    stack = build_agent(
        llm=FakeLLM(script),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
        chunk_size=200,
    )
    app.state.agent = stack.agent
    app.state.pipeline = stack.pipeline
    return TestClient(app, raise_server_exceptions=False)


# ---------- /healthz ----------

def test_healthz_ok():
    c = _client()
    r = c.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["checks"]["agent"] is True
    assert body["checks"]["pipeline"] is True


def test_healthz_degraded_when_agent_missing():
    from fastapi import FastAPI

    from app.api.middleware import CorrelationIdMiddleware
    from app.api.routes import router

    app = FastAPI()
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)
    # no app.state.agent
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "degraded"


# ---------- /ingest ----------

def test_ingest_ok():
    c = _client()
    payload = base64.b64encode(b"hello world from M3.4").decode()
    r = c.post("/ingest", json={"content_base64": payload, "source": "a.txt"})
    assert r.status_code == 200
    body = r.json()
    assert body["skipped"] is False
    assert body["n_chunks"] >= 1
    assert body["doc_id"]


def test_ingest_idempotent():
    c = _client()
    payload = base64.b64encode(b"hello").decode()
    r1 = c.post("/ingest", json={"content_base64": payload}).json()
    r2 = c.post("/ingest", json={"content_base64": payload}).json()
    assert r1["doc_id"] == r2["doc_id"]
    assert r2["skipped"] is True


def test_ingest_bad_base64_422():
    c = _client()
    r = c.post("/ingest", json={"content_base64": "!!!not base64!!!"})
    assert r.status_code == 422


def test_ingest_empty_content_422():
    c = _client()
    r = c.post("/ingest", json={"content_base64": ""})
    assert r.status_code == 422


# ---------- /query ----------

def test_query_ok():
    script = [
        '{"action":"tool","tool":"calculator","args":{"expression":"2+3"}}',
        '{"action":"finish","final_answer":"5","confidence":"high"}',
    ]
    c = _client(script)
    r = c.post("/query", json={"question": "what is 2+3?"})
    assert r.status_code == 200
    body = r.json()
    assert body["output"]["final_answer"] == "5"
    assert body["output"]["confidence"] == "high"
    assert body["elapsed_ms"] >= 0
    assert "correlation_id" in body


def test_query_echoes_cid():
    c = _client()
    r = c.post(
        "/query",
        json={"question": "hi"},
        headers={REQUEST_ID_HEADER: "q-123"},
    )
    assert r.status_code == 200
    assert r.json()["correlation_id"] == "q-123"


def test_query_empty_question_422():
    c = _client()
    r = c.post("/query", json={"question": "   "})
    assert r.status_code == 422


def test_query_extra_field_422():
    c = _client()
    r = c.post("/query", json={"question": "hi", "junk": 1})
    assert r.status_code == 422


# ---------- /metrics ----------

def test_metrics_shape():
    c = _client()
    # trigger some activity
    c.post("/ingest", json={"content_base64": base64.b64encode(b"x").decode()})
    c.post("/query", json={"question": "hi"})
    r = c.get("/metrics")
    assert r.status_code == 200
    snap = r.json()
    assert "counters" in snap
    assert "histograms" in snap
    assert snap["counters"].get("ingest.requests", 0) >= 1
    assert snap["counters"].get("query.requests", 0) >= 1
    assert "query.latency_ms" in snap["histograms"]


def test_metrics_refused_counter():
    script = [
        (
            '{"action":"finish","final_answer":"cannot","refused":true,'
            '"reason":"no evidence","confidence":"low"}'
        )
    ]
    c = _client(script)
    c.post("/query", json={"question": "q"})
    snap = c.get("/metrics").json()
    assert snap["counters"].get("query.refused", 0) >= 1
