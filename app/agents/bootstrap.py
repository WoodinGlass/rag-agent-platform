"""Factory: wire RAG pipeline + tools + registry + agent in one place.

Pure assembly. No IO performed here beyond constructing the objects
the caller already chose (embedder, store, http fetch fn).

Agent backend selection (M6.1):
- `state_machine` (default): `app.agents.agent.Agent`
- `langgraph`:              `app.agents.langgraph_backend.LangGraphAgent`
Both satisfy `AgentBackend`.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from app.agents.backend import AgentBackend
from app.agents.registry import ToolRegistry
from app.rag.chunker import CHUNKER_VERSION
from app.rag.embedder import Embedder
from app.rag.pipeline import RagPipeline
from app.rag.store import VectorStore
from app.tools.adapters.http import HttpResponse
from app.tools.adapters.rag import make_retrieve_fn
from app.tools.calculator import make_calculator_tool
from app.tools.search_docs import make_search_docs_tool
from app.tools.web_fetch import make_web_fetch_tool


@dataclass
class AgentStack:
    agent: AgentBackend
    pipeline: RagPipeline
    registry: ToolRegistry


DEFAULT_MAX_STEPS = 6


def _build_agent(
    backend: str,
    llm: Any,
    registry: ToolRegistry,
    max_steps: int,
) -> AgentBackend:
    if backend == "state_machine":
        from app.agents.agent import Agent as StateMachineAgent

        return StateMachineAgent(llm=llm, registry=registry, max_steps=max_steps)
    if backend == "langgraph":
        from app.agents.langgraph_backend import LangGraphAgent

        return LangGraphAgent(llm=llm, registry=registry, max_steps=max_steps)
    raise ValueError(f"unknown agent backend: {backend!r}")


def build_agent(
    *,
    llm: Any,
    embedder: Embedder,
    store: VectorStore,
    http_fetch_fn: Callable[[str, int], HttpResponse] | None = None,
    chunk_size: int = 200,
    chunker_version: str = CHUNKER_VERSION,
    max_steps: int = DEFAULT_MAX_STEPS,
    backend: str = "state_machine",
) -> AgentStack:
    """Assemble an Agent with the standard 3 tools.

    `http_fetch_fn` is optional — if provided, `web_fetch` is registered.
    `backend` selects the agent loop implementation.
    """
    pipeline = RagPipeline(
        embedder,
        store,
        chunk_size=chunk_size,
        chunker_version=chunker_version,
    )

    registry = ToolRegistry()
    registry.register(make_search_docs_tool(make_retrieve_fn(pipeline)))
    registry.register(make_calculator_tool())
    if http_fetch_fn is not None:
        registry.register(make_web_fetch_tool(http_fetch_fn))

    agent = _build_agent(backend, llm, registry, max_steps)
    return AgentStack(agent=agent, pipeline=pipeline, registry=registry)
