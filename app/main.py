"""Application entrypoint.

M3.3: minimal — install middleware, expose /healthz (shallow).
M3.4/M3.5 will add routes and wire the agent stack via lifespan.
"""
from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api.middleware import CorrelationIdMiddleware
from app.api.schemas import ErrorResponse, HealthResponse
from app.core.config import get_settings
from app.core.logging import configure_logging, new_correlation_id

APP_VERSION = "0.1.0"


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)

    app = FastAPI(
        title="rag-agent-platform",
        version=APP_VERSION,
        docs_url="/docs",
        redoc_url=None,
    )
    app.add_middleware(CorrelationIdMiddleware)

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", version=APP_VERSION, uptime_s=0.0)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
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
