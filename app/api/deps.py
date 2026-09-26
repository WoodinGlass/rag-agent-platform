"""FastAPI dependencies.

Wiring lives here, not in global mutable state. Routes declare what they
need via `Depends(...)` and get it. Swapping implementations (FakeLLM ->
real LLM) is a one-line change here, not a refactor across routes.

The heavy lifting (building the AgentStack) is cached at app startup via
`app.state` — see `app/main.py` (M3.5).
"""
from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from app.agents.agent import Agent
from app.core.config import Settings, get_settings
from app.core.logging import get_correlation_id
from app.rag.pipeline import RagPipeline

# ---------- basic accessors ----------


def get_app_settings() -> Settings:
    return get_settings()


def get_correlation_id_dep() -> str:
    """Correlation id set by middleware (M3.3). Empty if middleware off."""
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


# ---------- typed aliases for cleaner signatures ----------

SettingsDep = Annotated[Settings, Depends(get_app_settings)]
CorrelationIdDep = Annotated[str, Depends(get_correlation_id_dep)]
AgentDep = Annotated[Agent, Depends(get_agent)]
PipelineDep = Annotated[RagPipeline, Depends(get_pipeline)]
