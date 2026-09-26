"""Dependency wiring tests.

Uses a minimal FastAPI app (no routes yet) to verify Depends() resolution
and error behavior. Kept offline; no network, no LLM.

Uses `Annotated[..., Depends(...)]` so ruff B008 doesn't flag argument
defaults (FastAPI's classic pattern).
"""
from typing import Annotated

import pytest
from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient

from app.agents.agent import Agent
from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.api.deps import get_agent, get_correlation_id_dep, get_pipeline
from app.core.config import Settings
from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore

AgentAnnotated = Annotated[Agent, Depends(get_agent)]
PipelineAnnotated = Annotated[RagPipeline, Depends(get_pipeline)]


def _stack():
    return build_agent(
        llm=FakeLLM(['{"action":"finish","final_answer":"ok"}']),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )


def _app_with_state(agent=None, pipeline=None) -> FastAPI:
    app = FastAPI()
    if agent is not None:
        app.state.agent = agent
    if pipeline is not None:
        app.state.pipeline = pipeline
    return app


def test_get_agent_returns_app_state():
    stack = _stack()
    app = _app_with_state(agent=stack.agent, pipeline=stack.pipeline)

    @app.get("/probe")
    def probe(a: AgentAnnotated):
        return {"ok": a is stack.agent}

    client = TestClient(app)
    r = client.get("/probe")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_get_pipeline_returns_app_state():
    stack = _stack()
    app = _app_with_state(agent=stack.agent, pipeline=stack.pipeline)

    @app.get("/probe")
    def probe(p: PipelineAnnotated):
        return {"ok": isinstance(p, RagPipeline)}

    client = TestClient(app)
    r = client.get("/probe")
    assert r.status_code == 200
    assert r.json() == {"ok": True}


def test_get_agent_raises_when_state_missing():
    app = FastAPI()  # no state.agent

    @app.get("/probe")
    def probe(a: AgentAnnotated):
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=True)
    with pytest.raises(RuntimeError, match="app.state.agent not initialized"):
        client.get("/probe")


def test_get_pipeline_raises_when_state_missing():
    app = FastAPI()

    @app.get("/probe")
    def probe(p: PipelineAnnotated):
        return {"ok": True}

    client = TestClient(app, raise_server_exceptions=True)
    with pytest.raises(RuntimeError, match="app.state.pipeline not initialized"):
        client.get("/probe")


def test_get_settings_returns_settings():
    from app.api.deps import get_app_settings

    s = get_app_settings()
    assert isinstance(s, Settings)


def test_correlation_id_default_empty():
    # no middleware attached in this test -> empty string
    assert get_correlation_id_dep() == ""
