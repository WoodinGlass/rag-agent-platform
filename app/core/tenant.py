"""Tenant context for the current request.

Why a contextvar: the agent loop calls `search_docs` deep in the call
stack (LLM -> tool -> adapter -> pipeline). Plumbing `tenant_id` as a
parameter through every layer would leak an HTTP concern into the agent
and tool code. Instead:

- The HTTP layer (FastAPI dependency) sets the tenant at the start of
  the request.
- Anything downstream (routes, tools, adapters) reads it via
  `get_tenant_id()` without knowing where the request came from.
- The dependency resets it in a `finally` block, so tests and long-lived
  workers never leak tenant state across requests.

Defaults to None, which downstream code interprets as "single-tenant
mode" (no tenant filter).
"""
from __future__ import annotations

from contextvars import ContextVar, Token

_tenant_id: ContextVar[str | None] = ContextVar("tenant_id", default=None)


def get_tenant_id() -> str | None:
    """Return the tenant for the current execution context, or None."""
    return _tenant_id.get()


def set_tenant_id(tenant_id: str | None) -> Token:
    """Set the tenant; return a token for `reset_tenant_id`."""
    return _tenant_id.set(tenant_id)


def reset_tenant_id(token: Token) -> None:
    """Reset the tenant to the value before `set_tenant_id`."""
    _tenant_id.reset(token)
