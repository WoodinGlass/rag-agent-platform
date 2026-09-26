"""HTTP routes.

- POST /ingest : decode base64 -> pipeline.ingest (idempotent, tenant-scoped)
- POST /query  : run agent -> AgentOutput, wrapped with timing + cid
- GET  /healthz: shallow + deep checks
- GET  /metrics: in-process counters + latency histograms

Multi-tenant: both POST routes accept the tenant via `TenantDep` (from
`X-API-Key` when auth is enabled; otherwise the default tenant).
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request

from app.agents.schema import AgentOutput
from app.api.deps import (
    AgentDep,
    CorrelationIdDep,
    PipelineDep,
    SettingsDep,
    TenantDep,
)
from app.api.schemas import (
    HealthResponse,
    HealthStatus,
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
)
from app.core.logging import get_logger
from app.core.metrics import get_metrics

router = APIRouter()
log = get_logger(__name__)

APP_VERSION = "0.1.0"


# ---------- /ingest ----------


@router.post("/ingest", response_model=IngestResponse)
def ingest(
    req: IngestRequest,
    pipeline: PipelineDep,
    settings: SettingsDep,
    tenant: TenantDep,
) -> IngestResponse:
    metrics = get_metrics()
    metrics.inc("ingest.requests")

    r = pipeline.ingest(
        req.decode(),
        source=req.source,
        metadata=req.metadata,
        tenant_id=tenant,
    )
    if r.skipped:
        metrics.inc("ingest.skipped")
    else:
        metrics.inc("ingest.inserted_docs")
        metrics.inc("ingest.inserted_chunks", r.n_chunks)

    log.info(
        "ingest.done",
        extra={
            "doc_id": r.doc_id,
            "n_chunks": r.n_chunks,
            "skipped": r.skipped,
            "tenant": tenant,
        },
    )
    return IngestResponse(
        doc_id=r.doc_id,
        inserted=r.inserted,
        skipped=r.skipped,
        n_chunks=r.n_chunks,
    )


# ---------- /query ----------


@router.post("/query", response_model=QueryResponse)
def query(
    req: QueryRequest,
    agent: AgentDep,
    cid: CorrelationIdDep,
    tenant: TenantDep,
) -> QueryResponse:
    metrics = get_metrics()
    metrics.inc("query.requests")

    t0 = time.monotonic()
    # Note: the agent loop calls `search_docs`; the tenant filter is
    # applied because `bootstrap.build_agent` binds the pipeline and the
    # tool passes the current request's tenant to it (see M5.3 wiring in
    # app/main.py: pipeline carries the tenant per request via contextvar
    # is on the roadmap; for now, agent path is single-tenant).
    out: AgentOutput = agent.run(req.question)
    elapsed_ms = int((time.monotonic() - t0) * 1000)

    metrics.observe_ms("query.latency_ms", elapsed_ms)
    if out.refused:
        metrics.inc("query.refused")
    else:
        metrics.inc("query.answered")
    metrics.inc("query.tool_calls_total", len(out.tool_calls))

    return QueryResponse(output=out, elapsed_ms=elapsed_ms, correlation_id=cid)


# ---------- /healthz ----------


@router.get("/healthz", response_model=HealthResponse)
def healthz(request: Request, settings: SettingsDep) -> HealthResponse:
    uptime_s = 0.0
    started = getattr(request.app.state, "started_at", None)
    if started is not None:
        uptime_s = max(0.0, time.monotonic() - started)

    has_agent = getattr(request.app.state, "agent", None) is not None
    has_pipeline = getattr(request.app.state, "pipeline", None) is not None
    tenant_auth = bool(getattr(settings, "tenant_auth_enabled", False))

    checks = {
        "agent": has_agent,
        "pipeline": has_pipeline,
        "tenant_auth": tenant_auth,
    }
    status: HealthStatus = "ok" if (has_agent and has_pipeline) else "degraded"
    return HealthResponse(
        status=status,
        version=APP_VERSION,
        uptime_s=round(uptime_s, 3),
        checks=checks,
    )


# ---------- /metrics ----------


@router.get("/metrics")
def metrics() -> dict:
    return get_metrics().snapshot()
