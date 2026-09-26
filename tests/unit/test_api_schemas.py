import base64

import pytest
from pydantic import ValidationError

from app.agents.schema import AgentOutput
from app.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)

# ---------- IngestRequest ----------

def test_ingest_request_valid():
    payload = base64.b64encode(b"hello world").decode()
    req = IngestRequest(content_base64=payload, source="a.txt")
    assert req.decode() == b"hello world"
    assert req.source == "a.txt"


def test_ingest_request_metadata_defaults_empty():
    payload = base64.b64encode(b"x").decode()
    req = IngestRequest(content_base64=payload)
    assert req.metadata == {}
    assert req.source == ""


def test_ingest_request_invalid_base64_rejected():
    with pytest.raises(ValidationError, match="not valid base64"):
        IngestRequest(content_base64="!!!not-base64!!!")


def test_ingest_request_empty_rejected():
    with pytest.raises(ValidationError):
        IngestRequest(content_base64="")


def test_ingest_request_extra_field_rejected():
    payload = base64.b64encode(b"x").decode()
    with pytest.raises(ValidationError):
        IngestRequest(content_base64=payload, junk=1)  # type: ignore[call-arg]


def test_ingest_request_metadata_passes_through():
    payload = base64.b64encode(b"x").decode()
    req = IngestRequest(content_base64=payload, metadata={"corpus_id": "d1"})
    assert req.metadata["corpus_id"] == "d1"


# ---------- IngestResponse ----------

def test_ingest_response_valid():
    r = IngestResponse(doc_id="abc123", inserted=3, skipped=False, n_chunks=3)
    assert r.n_chunks == 3
    assert r.skipped is False


def test_ingest_response_negative_inserted_rejected():
    with pytest.raises(ValidationError):
        IngestResponse(doc_id="d", inserted=-1, skipped=False, n_chunks=0)


def test_ingest_response_empty_doc_id_rejected():
    with pytest.raises(ValidationError):
        IngestResponse(doc_id="", inserted=0, skipped=True, n_chunks=0)


# ---------- QueryRequest ----------

def test_query_request_valid():
    assert QueryRequest(question="what is RAG?").question == "what is RAG?"


def test_query_request_strips():
    assert QueryRequest(question="  hi  ").question == "hi"


def test_query_request_empty_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="")


def test_query_request_whitespace_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="   ")


def test_query_request_too_long_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="x" * 4001)


def test_query_request_extra_field_rejected():
    with pytest.raises(ValidationError):
        QueryRequest(question="q", junk=1)  # type: ignore[call-arg]


# ---------- QueryResponse ----------

def test_query_response_wraps_agent_output():
    out = AgentOutput(question="q", final_answer="a", confidence="high")
    resp = QueryResponse(output=out, elapsed_ms=42, correlation_id="cid123")
    assert resp.output.final_answer == "a"
    assert resp.elapsed_ms == 42
    assert resp.correlation_id == "cid123"


def test_query_response_defaults():
    out = AgentOutput(question="q", final_answer="a")
    resp = QueryResponse(output=out)
    assert resp.elapsed_ms == 0
    assert resp.correlation_id == ""


def test_query_response_negative_elapsed_rejected():
    out = AgentOutput(question="q", final_answer="a")
    with pytest.raises(ValidationError):
        QueryResponse(output=out, elapsed_ms=-1)


def test_query_response_roundtrip_json():
    out = AgentOutput(question="q", final_answer="a")
    resp = QueryResponse(output=out, elapsed_ms=1, correlation_id="c")
    raw = resp.model_dump_json()
    reparsed = QueryResponse.model_validate_json(raw)
    assert reparsed == resp


# ---------- HealthResponse ----------

def test_health_response_ok():
    h = HealthResponse(version="0.1.0", uptime_s=12.5)
    assert h.status == "ok"
    assert h.checks == {}


def test_health_response_with_checks():
    h = HealthResponse(
        status="degraded",
        version="0.1.0",
        uptime_s=1.0,
        checks={"store": True, "llm": False},
    )
    assert h.status == "degraded"
    assert h.checks["llm"] is False


def test_health_response_invalid_status_rejected():
    with pytest.raises(ValidationError):
        HealthResponse(status="broken", version="0.1.0", uptime_s=0.0)  # type: ignore[arg-type]


def test_health_response_negative_uptime_rejected():
    with pytest.raises(ValidationError):
        HealthResponse(version="0.1.0", uptime_s=-1.0)


# ---------- ErrorResponse ----------

def test_error_response_minimal():
    e = ErrorResponse(error="not_found")
    assert e.detail == ""
    assert e.correlation_id == ""


def test_error_response_full():
    e = ErrorResponse(error="bad_request", detail="missing field", correlation_id="cid")
    assert e.detail == "missing field"


def test_error_response_empty_error_rejected():
    with pytest.raises(ValidationError):
        ErrorResponse(error="")
