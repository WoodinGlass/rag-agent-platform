"""LangGraph backend parity with the state machine.

Everything here is offline. Skipped if `langgraph` is not installed
(run: pip install -e '.[langgraph]').
"""
import json

import pytest

# skip whole module if the optional dep is not present
pytest.importorskip("langgraph")

from app.agents.agent import Agent
from app.agents.backend import AgentBackend
from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.agents.schema import AgentOutput
from app.rag.embedder import FakeEmbedder
from app.rag.store import MemoryStore


def _step_tool(name: str, args: dict | None = None) -> str:
    return json.dumps(
        {"action": "tool", "tool": name, "args": args or {}, "thought": ""}
    )


def _step_finish(
    answer: str,
    cites: list | None = None,
    confidence: str = "medium",
    refused: bool = False,
    reason: str = "",
) -> str:
    return json.dumps(
        {
            "action": "finish",
            "final_answer": answer,
            "citations": cites or [],
            "confidence": confidence,
            "refused": refused,
            "reason": reason,
        }
    )


def _stack(script, backend):
    return build_agent(
        llm=FakeLLM(list(script)),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
        chunk_size=200,
        backend=backend,
    )


# ---------- protocol + basic ----------

def test_langgraph_agent_satisfies_protocol():
    stack = _stack([_step_finish("ok")], backend="langgraph")
    assert isinstance(stack.agent, AgentBackend)


def test_langgraph_single_finish():
    stack = _stack([_step_finish("42", confidence="high")], backend="langgraph")
    out = stack.agent.run("what?")
    assert isinstance(out, AgentOutput)
    assert out.final_answer == "42"
    assert out.confidence == "high"
    assert out.tool_calls == []


def test_langgraph_single_tool_then_finish():
    script = [
        _step_tool("calculator", {"expression": "2+3"}),
        _step_finish("5", confidence="high"),
    ]
    stack = _stack(script, backend="langgraph")
    out = stack.agent.run("what is 2+3?")
    assert out.final_answer == "5"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].tool == "calculator"
    assert out.tool_calls[0].ok is True


def test_langgraph_multihop():
    script = [
        _step_tool("search_docs", {"query": "cat", "k": 1}),
        _step_tool("calculator", {"expression": "3*7"}),
        _step_finish("21", confidence="high"),
    ]
    stack = _stack(script, backend="langgraph")
    out = stack.agent.run("q")
    assert [tc.tool for tc in out.tool_calls] == ["search_docs", "calculator"]
    assert out.final_answer == "21"


def test_langgraph_refused_path():
    script = [
        _step_finish(
            "cannot answer",
            refused=True,
            reason="no evidence",
            confidence="low",
        )
    ]
    stack = _stack(script, backend="langgraph")
    out = stack.agent.run("q")
    assert out.refused is True
    assert out.reason == "no evidence"


def test_langgraph_unknown_tool_recorded_not_fatal():
    script = [
        _step_tool("nonexistent", {}),
        _step_finish("recovered"),
    ]
    stack = _stack(script, backend="langgraph")
    out = stack.agent.run("q")
    assert out.tool_calls[0].ok is False
    assert "unknown tool" in (out.tool_calls[0].error or "")
    assert out.final_answer == "recovered"


def test_langgraph_tool_exception_recorded_not_fatal():
    from app.tools.base import Tool

    def _boom() -> None:
        raise RuntimeError("kaboom")

    stack = _stack(
        [_step_tool("boom", {}), _step_finish("recovered")],
        backend="langgraph",
    )
    stack.registry.register(
        Tool(
            name="boom",
            description="always fails",
            input_schema={"type": "object", "properties": {}},
            func=_boom,
        )
    )
    out = stack.agent.run("q")
    assert out.tool_calls[0].ok is False
    assert "kaboom" in (out.tool_calls[0].error or "")
    assert out.final_answer == "recovered"


def test_langgraph_invalid_json_recovers():
    script = ["not json", _step_finish("recovered")]
    stack = _stack(script, backend="langgraph")
    out = stack.agent.run("q")
    assert out.final_answer == "recovered"


def test_langgraph_max_steps_synthesizes_refusal():
    script = [_step_tool("calculator", {"expression": "1+1"})] * 5
    stack = _stack(script, backend="langgraph")
    # build with max_steps=2 to hit the cap fast
    from app.agents.bootstrap import build_agent

    stack = build_agent(
        llm=FakeLLM(script),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
        chunk_size=200,
        backend="langgraph",
        max_steps=2,
    )
    out = stack.agent.run("q")
    assert out.refused is True
    assert "max steps" in out.reason


def test_langgraph_empty_question_rejected():
    stack = _stack([_step_finish("ok")], backend="langgraph")
    with pytest.raises(ValueError, match="non-empty"):
        stack.agent.run("   ")


# ---------- parity with the state machine ----------

def test_backends_agree_on_simple_finish():
    script = [_step_finish("same", confidence="high")]
    sm = _stack(list(script), backend="state_machine").agent.run("q")
    lg = _stack(list(script), backend="langgraph").agent.run("q")
    assert sm.final_answer == lg.final_answer
    assert sm.confidence == lg.confidence
    assert sm.refused == lg.refused


def test_backends_agree_on_tool_then_finish():
    script = [
        _step_tool("calculator", {"expression": "3+4"}),
        _step_finish("7", confidence="high"),
    ]
    sm = _stack(list(script), backend="state_machine").agent.run("q")
    lg = _stack(list(script), backend="langgraph").agent.run("q")
    assert sm.final_answer == lg.final_answer
    assert [tc.tool for tc in sm.tool_calls] == [tc.tool for tc in lg.tool_calls]
    assert [tc.ok for tc in sm.tool_calls] == [tc.ok for tc in lg.tool_calls]


def test_backends_agree_on_refused():
    script = [
        _step_finish(
            "no", refused=True, reason="r", confidence="low"
        )
    ]
    sm = _stack(list(script), backend="state_machine").agent.run("q")
    lg = _stack(list(script), backend="langgraph").agent.run("q")
    assert sm.refused == lg.refused
    assert sm.reason == lg.reason
    assert sm.confidence == lg.confidence


def test_backends_agree_on_citations():
    script = [
        _step_finish(
            "answer",
            cites=[{"doc_id": "d", "chunk_id": "d:0", "quote": "q"}],
        )
    ]
    sm = _stack(list(script), backend="state_machine").agent.run("q")
    lg = _stack(list(script), backend="langgraph").agent.run("q")
    assert len(sm.citations) == len(lg.citations) == 1
    assert sm.citations[0].quote == lg.citations[0].quote


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="unknown agent backend"):
        build_agent(
            llm=FakeLLM([_step_finish("x")]),
            embedder=FakeEmbedder(dim=32),
            store=MemoryStore(),
            backend="nope",
        )


def test_default_backend_is_state_machine():
    stack = build_agent(
        llm=FakeLLM([_step_finish("x")]),
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )
    # state machine Agent is not LangGraphAgent
    assert isinstance(stack.agent, Agent)
    from app.agents.langgraph_backend import LangGraphAgent

    assert not isinstance(stack.agent, LangGraphAgent)
