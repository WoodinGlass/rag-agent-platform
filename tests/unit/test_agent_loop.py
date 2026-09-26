import json

import pytest

from app.agents.agent import Agent, AgentError
from app.agents.llm import FakeLLM
from app.agents.registry import ToolRegistry
from app.agents.schema import AgentOutput
from app.rag.embedder import FakeEmbedder
from app.rag.pipeline import RagPipeline
from app.rag.store import MemoryStore
from app.tools.adapters.rag import make_retrieve_fn
from app.tools.base import Tool
from app.tools.calculator import make_calculator_tool
from app.tools.search_docs import make_search_docs_tool


def _registry() -> ToolRegistry:
    pipe = RagPipeline(FakeEmbedder(dim=64), MemoryStore(), chunk_size=200)
    pipe.ingest(b"The cat sat on the mat. Dogs bark.", source="a.txt")
    reg = ToolRegistry()
    reg.register(make_search_docs_tool(make_retrieve_fn(pipe)))
    reg.register(make_calculator_tool())
    return reg


def _step_tool(name: str, args: dict | None = None, thought: str = "") -> str:
    return json.dumps(
        {"action": "tool", "tool": name, "args": args or {}, "thought": thought}
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


def test_single_tool_then_finish():
    llm = FakeLLM(
        [
            _step_tool("calculator", {"expression": "2+3"}),
            _step_finish("5", confidence="high"),
        ]
    )
    out = Agent(llm, _registry()).run("what is 2+3?")
    assert isinstance(out, AgentOutput)
    assert out.final_answer == "5"
    assert out.confidence == "high"
    assert len(out.tool_calls) == 1
    assert out.tool_calls[0].tool == "calculator"
    assert out.tool_calls[0].ok is True


def test_multi_step_search_then_calculator():
    llm = FakeLLM(
        [
            _step_tool("search_docs", {"query": "cat", "k": 1}),
            _step_tool("calculator", {"expression": "3*7"}),
            _step_finish("21", confidence="high"),
        ]
    )
    out = Agent(llm, _registry()).run("how many?")
    assert [tc.tool for tc in out.tool_calls] == ["search_docs", "calculator"]
    assert out.final_answer == "21"


def test_llm_sees_tool_result_between_steps():
    llm = FakeLLM(
        [
            _step_tool("calculator", {"expression": "2+3"}),
            _step_finish("5"),
        ]
    )
    Agent(llm, _registry()).run("q")
    second_call = llm.calls[1]
    tool_msgs = [m for m in second_call if "Tool result" in m.get("content", "")]
    assert tool_msgs, "agent must feed tool result back"
    assert '"result": 5' in tool_msgs[0]["content"]


def test_unknown_tool_is_recorded_not_fatal():
    llm = FakeLLM(
        [
            _step_tool("nonexistent", {}),
            _step_finish("done"),
        ]
    )
    out = Agent(llm, _registry()).run("q")
    assert out.tool_calls[0].ok is False
    assert "unknown tool" in (out.tool_calls[0].error or "")
    assert out.final_answer == "done"


def test_tool_exception_is_recorded_not_fatal():
    def _boom() -> None:
        raise RuntimeError("kaboom")

    reg = _registry()
    reg.register(
        Tool(
            name="boom",
            description="always fails",
            input_schema={"type": "object", "properties": {}},
            func=_boom,
        )
    )
    llm = FakeLLM(
        [
            _step_tool("boom", {}),
            _step_finish("recovered"),
        ]
    )
    out = Agent(llm, reg).run("q")
    assert out.tool_calls[0].ok is False
    assert "kaboom" in (out.tool_calls[0].error or "")
    assert out.final_answer == "recovered"


def test_invalid_json_feeds_back_and_recovers():
    llm = FakeLLM(["this is not json", _step_finish("recovered")])
    out = Agent(llm, _registry()).run("q")
    assert out.final_answer == "recovered"
    feedback = [
        m for m in llm.calls[1] if "invalid" in m.get("content", "").lower()
    ]
    assert feedback


def test_max_steps_exceeded():
    llm = FakeLLM([_step_tool("calculator", {"expression": "1+1"})] * 5)
    with pytest.raises(AgentError, match="max steps"):
        Agent(llm, _registry(), max_steps=3).run("q")


def test_refused_path():
    llm = FakeLLM(
        [
            _step_finish(
                "cannot answer",
                refused=True,
                reason="no evidence",
                confidence="low",
            )
        ]
    )
    out = Agent(llm, _registry()).run("q")
    assert out.refused is True
    assert out.reason == "no evidence"
    assert out.confidence == "low"


def test_citations_passed_through():
    llm = FakeLLM(
        [
            _step_tool("search_docs", {"query": "cat", "k": 1}),
            _step_finish(
                "The cat sat on the mat.",
                cites=[{"doc_id": "d1", "chunk_id": "d1:0000", "quote": "cat"}],
                confidence="high",
            ),
        ]
    )
    out = Agent(llm, _registry()).run("where did the cat sit?")
    assert len(out.citations) == 1
    assert out.citations[0].quote == "cat"


def test_empty_question_rejected():
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="non-empty"):
        Agent(llm, _registry()).run("   ")
