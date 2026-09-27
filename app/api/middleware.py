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

from app.api.limits import (
    TokenBucketLimiter,
    check_content_length,
)
from app.core.auth import AuthError, resolve_tenant
from app.core.config import get_settings
from app.core.logging import (
    get_correlation_id,
    get_logger,
    set_correlation_id,
)
from app.core.tenant import set_tenant_id

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

# Paths that must stay reachable without auth (monitoring, docs).
_AUTH_EXEMPT_PATHS = {"/healthz", "/metrics", "/docs", "/redoc", "/openapi.json"}


class TenantMiddleware(BaseHTTPMiddleware):
    """Resolve tenant from X-API-Key and expose it via contextvar.

    Runs early in the stack, alongside CorrelationIdMiddleware. Because
    it lives in the async middleware layer, the contextvar set here is
    visible to sync route handlers (Starlette copies the context when
    running sync endpoints in the threadpool).

    - Auth is enabled -> require X-API-Key, else 401.
    - Auth is disabled -> set the default tenant.
    - Monitoring paths (`/healthz`, `/metrics`, `/docs`) are always
      reachable so probes and dashboards never need a key.
    """

    async def dispatch(self, request: Request, call_next):
        # Prefer app.state.settings (set in create_app and in test apps)
        # so dependency overrides in tests are honored. Fallback to the
        # module singleton for apps that don't set it.
        settings = getattr(request.app.state, "settings", None) or get_settings()

        if request.url.path in _AUTH_EXEMPT_PATHS:
            # Still set a tenant so downstream code doesn't see None.
            set_tenant_id(settings.default_tenant)
            return await call_next(request)

        x_api_key = request.headers.get("X-API-Key")
        try:
            tenant = resolve_tenant(x_api_key, settings)
        except AuthError as e:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "unauthorized",
                    "detail": str(e),
                    "correlation_id": get_correlation_id(),
                },
            )
        set_tenant_id(tenant)
        return await call_next(request)

class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token-bucket rate limit keyed by tenant (X-API-Key) or client IP.

    Reads settings from `request.app.state.settings` (falling back to the
    module singleton) so test apps can override without touching globals.
    The limiter itself lives on `app.state.rate_limiter`, created in
    `create_app` when `RATE_LIMIT_ENABLED=true`. Disabled -> pass-through.

    Exempt paths (`/healthz`, `/metrics`, `/docs`) never rate limit so
    probes and dashboards are unaffected.

    On rejection: 429 with `Retry-After` (integer seconds) and the
    standard error body. Preserves the correlation id.
    """

    async def dispatch(self, request: Request, call_next):
        settings = getattr(request.app.state, "settings", None) or get_settings()
        if not getattr(settings, "rate_limit_enabled", False):
            return await call_next(request)

        if request.url.path in _AUTH_EXEMPT_PATHS:
            return await call_next(request)

        limiter: TokenBucketLimiter | None = getattr(
            request.app.state, "rate_limiter", None
        )
        if limiter is None:
            return await call_next(request)

        api_key = request.headers.get("X-API-Key")
        client_ip = request.client.host if request.client else "unknown"
        key = api_key or client_ip

        allowed, retry_after = limiter.allow(key)
        if allowed:
            return await call_next(request)

        # Round up so the client does not retry early.
        retry_after_s = max(1, int(retry_after + 0.5))
        return JSONResponse(
            status_code=429,
            headers={"Retry-After": str(retry_after_s)},
            content={
                "error": "rate_limited",
                "detail": f"retry after {retry_after_s}s",
                "retry_after_s": retry_after_s,
                "correlation_id": get_correlation_id(),
            },
        )


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized requests from the Content-Length header.

    A fast header-only check. Chunked uploads (no Content-Length) are
    allowed through — a streaming cap is a follow-up (documented in
    `docs/limitations.md`). Disabled when `MAX_BODY_SIZE_BYTES=0`.
    """

    async def dispatch(self, request: Request, call_next):
        settings = getattr(request.app.state, "settings", None) or get_settings()
        max_bytes = int(getattr(settings, "max_body_size_bytes", 0))
        if max_bytes <= 0:
            return await call_next(request)

        verdict = check_content_length(
            request.headers.get("content-length"), max_bytes
        )
        if verdict.allowed:
            return await call_next(request)

        return JSONResponse(
            status_code=413,
            content={
                "error": "payload_too_large",
                "detail": verdict.reason,
                "correlation_id": get_correlation_id(),
                **verdict.field,
            },
        )
