"""FastAPI dependencies.

Wiring lives here, not in global mutable state. Routes declare what they
need via `Annotated[..., Depends(...)]` and get it.

The heavy lifting (building the AgentStack) is cached at app startup via
`app.state` — see `app/main.py`.

Tenant (M6.2): `get_tenant_id` is a yield-dependency. It resolves the
tenant from `X-API-Key`, sets it on a contextvar, yields the value, and
resets the contextvar in a `finally`. That makes the tenant visible to
everything downstream — including tools called from inside the agent
loop — without threading a parameter through every function.
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agents.agent import Agent
from app.core.config import Settings, get_settings
from app.core.logging import get_correlation_id
from app.core.tenant import get_tenant_id as get_current_tenant
from app.rag.pipeline import RagPipeline

# ---------- basic accessors ----------


def get_app_settings() -> Settings:
    return get_settings()


def get_correlation_id_dep() -> str:
    """Correlation id set by middleware. Empty if middleware off."""
    return get_correlation_id()


# ---------- stack accessors (from app.state, set in main.py) ----------


def get_agent(request: Request) -> Agent:
    """Agent built once at app startup and stored on app.state."""
    agent = getattr(request.app.state, "agent", None)
    if agent is None:
        raise RuntimeError("app.state.agent not initialized")
    return agent


def get_pipeline(request: Request) -> RagPipeline:
    """RAG pipeline built once at app startup."""
    pipeline = getattr(request.app.state, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("app.state.pipeline not initialized")
    return pipeline


# ---------- auth / tenant ----------


def get_tenant_id(
    settings: Annotated[Settings, Depends(get_app_settings)],
) -> str:
    """Return the tenant resolved by TenantMiddleware.

    Fallback: if middleware is not installed (e.g. minimal test app),
    return the configured default tenant.
    """
    tenant = get_current_tenant()
    return tenant if tenant is not None else settings.default_tenant


# ---------- typed aliases for cleaner signatures ----------

SettingsDep = Annotated[Settings, Depends(get_app_settings)]
CorrelationIdDep = Annotated[str, Depends(get_correlation_id_dep)]
AgentDep = Annotated[Agent, Depends(get_agent)]
PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
TenantDep = Annotated[str, Depends(get_tenant_id)]
