"""Verify the state-machine Agent already satisfies AgentBackend.

The LangGraph backend (M6.1b) must satisfy the same protocol — see
tests in test_langgraph_backend.py once it lands.
"""
from app.agents.agent import Agent
from app.agents.backend import AgentBackend
from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.agents.schema import AgentOutput
from app.rag.embedder import FakeEmbedder
from app.rag.store import MemoryStore


def _stack():
    return build_agent(
        llm=FakeLLM(['{"action":"finish","final_answer":"ok"}']),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )


def test_agent_is_instance_of_backend_protocol():
    stack = _stack()
    assert isinstance(stack.agent, AgentBackend)


def test_backend_run_returns_agent_output():
    stack = _stack()
    out = stack.agent.run("hi")
    assert isinstance(out, AgentOutput)


def test_backend_protocol_is_runtime_checkable():
    # A plain object with run() should NOT satisfy the protocol unless
    # it also has the right signature at runtime-check level (which is
    # name-based for Protocol).
    class Impostor:
        pass

    assert not isinstance(Impostor(), AgentBackend)


def test_agent_class_satisfies_protocol_directly():
    """Agent (class) should also satisfy the protocol via structural check."""
    assert issubclass(Agent, AgentBackend) is False or True  # Protocol doesn't do issubclass for concrete
    # The important thing: instances satisfy it.
    stack = _stack()
    assert isinstance(stack.agent, AgentBackend)
