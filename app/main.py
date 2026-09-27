"""Application entrypoint.

- Build the agent stack once at startup (lifespan).
- Install correlation-id middleware.
- Mount routes from app.api.routes.
- Error handler keeps ErrorResponse contract on 500.
- Provider selected from settings.llm_provider:
  fake (default) | openai | groq | anthropic.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.agents.bootstrap import AgentStack, build_agent
from app.agents.llm import (
    GROQ_BASE_URL,
    GROQ_DEFAULT_MODEL,
    AnthropicLLM,
    FakeLLM,
    OpenAILLM,
)
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


def _build_llm(settings: Settings):
    """Select the LLM backend from settings.llm_provider."""
    provider = settings.llm_provider

    if provider == "fake":
        return FakeLLM([_NO_PROVIDER_REPLY])

    if provider == "openai":
        return OpenAILLM(
            model=settings.llm_model,
            api_key=settings.openai_api_key or None,
            base_url=settings.llm_base_url or None,
            timeout_s=float(settings.llm_timeout_s),
        )

    if provider == "groq":
        return OpenAILLM(
            model=settings.llm_model or GROQ_DEFAULT_MODEL,
            api_key=settings.groq_api_key or None,
            base_url=settings.llm_base_url or GROQ_BASE_URL,
            timeout_s=float(settings.llm_timeout_s),
            # Groq free tier caps OTPM at 1000; keep requests small.
            max_tokens=256,
        )

    if provider == "anthropic":
        return AnthropicLLM(
            model=settings.llm_model,
            api_key=settings.anthropic_api_key or None,
        )

    raise ValueError(f"unknown llm_provider: {provider!r}")


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
    elif settings.vector_backend == "qdrant":
        store = get_store(
            "qdrant",
            url=settings.qdrant_url,
            collection=settings.qdrant_collection,
            dim=settings.qdrant_dim,
            api_key=settings.qdrant_api_key or None,
        )
    else:
        store = get_store("memory")

    llm = _build_llm(settings)

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
        exporter=settings.otel_exporter,
        otlp_endpoint=settings.otel_otlp_endpoint,
    )
    app.state.started_at = time.monotonic()
    app.state.stack = _build_stack(settings)
    app.state.agent = app.state.stack.agent
    app.state.pipeline = app.state.stack.pipeline
    yield


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="rag-agent-platform",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url=None,
        lifespan=lifespan,
    )
    # Expose settings on app.state so TenantMiddleware can see them.
    app.state.settings = settings
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
