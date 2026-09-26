"""Factory: wire RAG pipeline + tools + registry + agent in one place.

Pure assembly. No IO performed here beyond constructing the objects
the caller already chose (embedder, store, http fetch fn).

Usage:
    stack = build_agent(
        llm=FakeLLM(script),
        embedder=FakeEmbedder(dim=64),
        store=MemoryStore(),
    )
    out = stack.agent.run("...")
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from app.agents.agent import DEFAULT_MAX_STEPS, Agent
from app.agents.llm import LLM
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
    agent: Agent
    pipeline: RagPipeline
    registry: ToolRegistry


def build_agent(
    *,
    llm: LLM,
    embedder: Embedder,
    store: VectorStore,
    http_fetch_fn: Callable[[str, int], HttpResponse] | None = None,
    chunk_size: int = 200,
    chunker_version: str = CHUNKER_VERSION,
    max_steps: int = DEFAULT_MAX_STEPS,
) -> AgentStack:
    """Assemble an Agent with the standard 3 tools.

    `http_fetch_fn` is optional — if provided, `web_fetch` is registered.
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

    agent = Agent(llm=llm, registry=registry, max_steps=max_steps)
    return AgentStack(agent=agent, pipeline=pipeline, registry=registry)
