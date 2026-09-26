import json

import pytest
from pydantic import ValidationError

from app.agents.schema import AgentOutput, Citation, ToolCall

# ---------- happy path ----------

def test_minimal_valid():
    out = AgentOutput(question="what?", final_answer="42")
    assert out.citations == []
    assert out.tool_calls == []
    assert out.confidence == "medium"
    assert out.refused is False


def test_with_citations_and_tool_calls():
    out = AgentOutput(
        question="who wrote 1984?",
        final_answer="George Orwell.",
        citations=[Citation(doc_id="d1", chunk_id="d1:0000", quote="Orwell")],
        tool_calls=[ToolCall(tool="search_docs", args={"query": "1984"})],
        confidence="high",
    )
    assert len(out.citations) == 1
    assert out.tool_calls[0].tool == "search_docs"
    assert out.confidence == "high"


def test_roundtrip_json():
    out = AgentOutput(
        question="q",
        final_answer="a",
        citations=[Citation(doc_id="d", chunk_id="d:0")],
    )
    raw = out.to_json()
    parsed = AgentOutput.from_json(raw)
    assert parsed == out


# ---------- validators ----------

def test_empty_question_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="", final_answer="x")


def test_whitespace_question_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="   ", final_answer="x")


def test_empty_final_answer_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="q", final_answer="")


def test_whitespace_final_answer_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="q", final_answer="   \n  ")


def test_final_answer_stripped():
    out = AgentOutput(question="q", final_answer="  hello  ")
    assert out.final_answer == "hello"


def test_question_stripped():
    out = AgentOutput(question="  q?  ", final_answer="a")
    assert out.question == "q?"


def test_extra_field_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="q", final_answer="a", unknown=1)  # type: ignore[call-arg]


def test_invalid_confidence_rejected():
    with pytest.raises(ValidationError):
        AgentOutput(question="q", final_answer="a", confidence="veryhigh")  # type: ignore[arg-type]


# ---------- refused semantics ----------

def test_refused_requires_reason():
    with pytest.raises(ValidationError, match="reason is required"):
        AgentOutput(
            question="q",
            final_answer="cannot answer",
            refused=True,
            reason="",
        )


def test_refused_with_reason_ok():
    out = AgentOutput(
        question="q",
        final_answer="cannot answer",
        refused=True,
        reason="no evidence in corpus",
    )
    assert out.refused is True
    assert out.reason == "no evidence in corpus"


def test_non_refused_reason_optional():
    out = AgentOutput(question="q", final_answer="a", reason="")
    assert out.refused is False
    assert out.reason == ""


# ---------- nested schemas ----------

def test_citation_extra_field_rejected():
    with pytest.raises(ValidationError):
        Citation(doc_id="d", chunk_id="c", unknown=1)  # type: ignore[call-arg]


def test_citation_empty_doc_rejected():
    with pytest.raises(ValidationError):
        Citation(doc_id="", chunk_id="c")


def test_tool_call_negative_latency_rejected():
    with pytest.raises(ValidationError):
        ToolCall(tool="x", latency_ms=-1)


def test_tool_call_defaults():
    tc = ToolCall(tool="calculator")
    assert tc.args == {}
    assert tc.ok is True
    assert tc.error is None
    assert tc.latency_ms == 0


# ---------- JSON parsing from LLM ----------

def test_from_json_accepts_plain_object():
    raw = json.dumps({"question": "q", "final_answer": "a"})
    out = AgentOutput.from_json(raw)
    assert out.final_answer == "a"


def test_from_json_rejects_missing_required():
    raw = json.dumps({"question": "q"})
    with pytest.raises(ValidationError):
        AgentOutput.from_json(raw)


def test_from_json_rejects_extra():
    raw = json.dumps({"question": "q", "final_answer": "a", "junk": 1})
    with pytest.raises(ValidationError):
        AgentOutput.from_json(raw)
