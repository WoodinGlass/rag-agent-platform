"""Application entrypoint.

- Build the agent stack once at startup (lifespan).
- Install correlation-id middleware.
- Mount routes from app.api.routes.
- Error handler keeps ErrorResponse contract on 500.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.bootstrap import AgentStack, build_agent
from app.agents.llm import FakeLLM
from app.api.middleware import CorrelationIdMiddleware, TenantMiddleware
from app.api.routes import router
from app.api.schemas import ErrorResponse
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, new_correlation_id
from app.core.tracing import configure_tracing
from app.rag.embedder import get_embedder
from app.rag.store import get_store

APP_VERSION = "0.1.0"

_NO_PROVIDER_REPLY = (
    '{"action": "finish", '
    '"final_answer": "No LLM provider configured; '
    'set LLM_PROVIDER to enable answers.", '
    '"confidence": "low", "refused": true, '
    '"reason": "no provider configured"}'
)


def _build_stack(settings: Settings) -> AgentStack:
    """Pick implementations from settings. Default = offline & deterministic."""
    if settings.embedder_provider == "fake":
        embedder = get_embedder("fake", dim=settings.embedder_dim)
    else:
        embedder = get_embedder(settings.embedder_provider)

    if settings.vector_backend == "chroma":
        store = get_store(
            "chroma",
            path=settings.chroma_path,
            collection=settings.chroma_collection,
        )
    else:
        store = get_store("memory")

    # MVP: FakeLLM scripted to answer immediately. M3.5+ will wire real
    # provider when llm_provider != "fake".
    llm = FakeLLM([_NO_PROVIDER_REPLY])

    return build_agent(
        llm=llm,
        embedder=embedder,
        store=store,
        chunk_size=settings.chunk_size,
        backend=settings.agent_backend,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    configure_tracing(
        enabled=settings.otel_enabled,
        service_name=settings.otel_service_name,
        console_exporter=settings.otel_console_exporter,
    )
    app.state.started_at = time.monotonic()
    app.state.stack = _build_stack(settings)
    app.state.agent = app.state.stack.agent
    app.state.pipeline = app.state.stack.pipeline
    yield
    # shutdown hooks (none yet)


def create_app() -> FastAPI:
    app = FastAPI(
        title="rag-agent-platform",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    # order: outermost runs first. CID first (so all logs have it),
    # then tenant (so auth rejection logs carry a cid).
    app.add_middleware(TenantMiddleware)
    app.add_middleware(CorrelationIdMiddleware)
    app.include_router(router)

    @app.exception_handler(Exception)
    async def _unhandled(_request: Request, exc: Exception) -> JSONResponse:
        cid = new_correlation_id()
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="internal_error",
                detail=str(exc),
                correlation_id=cid,
            ).model_dump(),
        )

    return app


app = create_app()
