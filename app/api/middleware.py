"""Correlation-ID + structured access-log middleware.

Behavior:
- Reads `X-Request-ID` from the incoming request; generates one if absent.
- Sets it in the contextvar from `app.core.logging` so every log line
  inside the request carries `cid`.
- Echoes it back in the `X-Request-ID` response header.
- Emits one JSON access log per request (method, path, status, dur_ms, cid).
- On unhandled exceptions, returns a JSON 500 with the cid header so the
  client can correlate the failure with server logs. If the app has its
  own exception handler, that runs first (below this middleware) and this
  block is not reached.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.logging import (
    get_correlation_id,
    get_logger,
    set_correlation_id,
)

log = get_logger("app.api.access")

REQUEST_ID_HEADER = "X-Request-ID"
MAX_CID_LEN = 64

# Paths that shouldn't spam the access log.
_QUIET_PATHS = {"/healthz", "/metrics", "/favicon.ico"}


def _clean_cid(raw: str | None) -> str:
    """Accept a client-provided id if it's a reasonable token; else empty."""
    if not raw:
        return ""
    raw = raw.strip()
    if not raw or len(raw) > MAX_CID_LEN:
        return ""
    for ch in raw:
        if not (ch.isalnum() or ch in "-_."):
            return ""
    return raw


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Set/echo correlation id + emit one structured access log per request."""

    def __init__(self, app, *, log_quiet_paths: bool = False) -> None:
        super().__init__(app)
        self.log_quiet_paths = log_quiet_paths

    async def dispatch(self, request: Request, call_next):
        incoming = _clean_cid(request.headers.get(REQUEST_ID_HEADER))
        cid = incoming or uuid.uuid4().hex[:16]
        set_correlation_id(cid)

        start = time.monotonic()
        try:
            response: Response = await call_next(request)
        except Exception as exc:
            dur_ms = int((time.monotonic() - start) * 1000)
            log.exception(
                "request.error",
                extra={
                    "cid": cid,
                    "method": request.method,
                    "path": request.url.path,
                    "dur_ms": dur_ms,
                    "client": request.client.host if request.client else "",
                },
            )
            response = JSONResponse(
                status_code=500,
                content={
                    "error": "internal_error",
                    "detail": str(exc),
                    "correlation_id": cid,
                },
            )
            response.headers[REQUEST_ID_HEADER] = cid
            return response

        dur_ms = int((time.monotonic() - start) * 1000)

        should_log = self.log_quiet_paths or request.url.path not in _QUIET_PATHS
        if should_log:
            log.info(
                "request.ok",
                extra={
                    "cid": cid,
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "dur_ms": dur_ms,
                    "client": request.client.host if request.client else "",
                },
            )

        # Use whatever cid is live now (inner code may have replaced it).
        response.headers[REQUEST_ID_HEADER] = get_correlation_id() or cid
        return response
