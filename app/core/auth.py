"""Tenant authentication.

Design:
- Auth is opt-in (`TENANT_AUTH_ENABLED`). When off, every request gets
  `DEFAULT_TENANT` and behaviour is identical to a single-tenant deploy.
- When on, requests must carry `X-API-Key`. The key maps to a tenant id.
- The mapping is a simple comma-separated string in `TENANT_KEYS`
  (`key1:tenantA,key2:tenantB`). This is deliberately small — a real
  deployment would read from a secrets store or DB. The parse/resolve
  split keeps that swap easy.
- Tenant id is the *only* thing routes need. It is treated as opaque.
"""
from __future__ import annotations

from app.core.config import Settings


class AuthError(RuntimeError):
    """Raised when a request cannot be attributed to a tenant."""


def parse_tenant_keys(raw: str) -> dict[str, str]:
    """Parse `key1:tenantA,key2:tenantB` -> {key1: tenantA, key2: tenantB}.

    Whitespace is stripped. Empty entries are skipped. Duplicate keys are
    rejected — this is a config error, not a runtime one.
    """
    out: dict[str, str] = {}
    if not raw:
        return out
    for entry in raw.split(","):
        entry = entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise AuthError(f"invalid tenant key entry: {entry!r}")
        key, _, tenant = entry.partition(":")
        key = key.strip()
        tenant = tenant.strip()
        if not key or not tenant:
            raise AuthError(f"invalid tenant key entry: {entry!r}")
        if key in out:
            raise AuthError(f"duplicate tenant key: {key!r}")
        out[key] = tenant
    return out


def resolve_tenant(api_key: str | None, settings: Settings) -> str:
    """Return the tenant id for this request.

    - Auth disabled -> default tenant.
    - Auth enabled + missing/invalid key -> AuthError (route maps to 401).
    """
    if not settings.tenant_auth_enabled:
        return settings.default_tenant

    if not api_key:
        raise AuthError("missing X-API-Key header")

    mapping = parse_tenant_keys(settings.tenant_keys)
    if not mapping:
        # Misconfigured server: auth on but no keys. Fail closed.
        raise AuthError("server has no tenant keys configured")

    tenant = mapping.get(api_key)
    if tenant is None:
        raise AuthError("invalid X-API-Key")
    return tenant
