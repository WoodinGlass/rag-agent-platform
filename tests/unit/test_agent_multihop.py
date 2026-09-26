"""Multi-hop test: agent must chain ≥2 tools and end with valid AgentOutput.

This is the M2 exit criteria, expressed as an offline test.
"""
import json

from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.agents.schema import AgentOutput
from app.rag.embedder import FakeEmbedder
from app.rag.store import MemoryStore

CORPUS = [
    ("cat.txt", b"The cat sat on the mat. The word cat has 3 letters."),
    ("dog.txt", b"Dogs bark loudly. The word dog has 3 letters."),
    ("quantum.txt", b"Quantum computing uses qubits and superposition."),
]


def _step_tool(name: str, args: dict) -> str:
    return json.dumps({"action": "tool", "tool": name, "args": args})


def _step_finish(answer: str, cites=None, confidence="medium", refused=False, reason="") -> str:
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


def _stack(script):
    llm = FakeLLM(script)
    stack = build_agent(
        llm=llm,
        embedder=FakeEmbedder(dim=64),
        store=MemoryStore(),
        chunk_size=200,
    )
    for name, data in CORPUS:
        stack.pipeline.ingest(data, source=name)
    return stack, llm


# ---------- multi-hop: search_docs -> calculator -> finish ----------

def test_multihop_search_then_calculate():
    script = [
        _step_tool("search_docs", {"query": "cat letters", "k": 2}),
        _step_tool("calculator", {"expression": "3*7"}),
        _step_finish("21", confidence="high"),
    ]
    stack, llm = _stack(script)

    out = stack.agent.run("How many letters in 'cat', multiplied by 7?")

    assert isinstance(out, AgentOutput)
    assert [tc.tool for tc in out.tool_calls] == ["search_docs", "calculator"]
    assert all(tc.ok for tc in out.tool_calls)
    assert out.final_answer == "21"
    assert out.confidence == "high"

    # LLM must have seen the tool results between steps
    last_msgs = llm.calls[-1]
    assert any("Tool result" in m.get("content", "") for m in last_msgs)


def test_multihop_uses_retrieved_citation():
    script = [
        _step_tool("search_docs", {"query": "cat", "k": 1}),
        _step_tool("calculator", {"expression": "3*7"}),
        _step_finish(
            "The word 'cat' has 3 letters; 3 x 7 = 21.",
            cites=[{"doc_id": "d", "chunk_id": "d:0000", "quote": "3 letters"}],
            confidence="high",
        ),
    ]
    stack, _ = _stack(script)
    out = stack.agent.run("q")

    assert len(out.citations) == 1
    assert out.citations[0].quote == "3 letters"
    # schema round-trip (M2.5 contract enforced end-to-end)
    assert AgentOutput.from_json(out.to_json()) == out


def test_multihop_schema_validation_passes():
    """Exit criteria: final output must satisfy AgentOutput."""
    script = [
        _step_tool("search_docs", {"query": "cat", "k": 1}),
        _step_finish("done", confidence="medium"),
    ]
    stack, _ = _stack(script)
    out = stack.agent.run("q")

    # If construction succeeded, schema passed. Double-check by re-parsing.
    reparsed = AgentOutput.model_validate_json(out.to_json())
    assert reparsed.question == out.question
    assert reparsed.final_answer == out.final_answer


# ---------- web_fetch registered when fn provided ----------

def test_web_fetch_registered_when_provided():
    from app.tools.adapters.http import HttpResponse

    def fake_fetch(url: str, max_bytes: int) -> HttpResponse:
        return HttpResponse(
            url=url,
            status=200,
            text="page body",
            headers={"content-type": "text/plain"},
            elapsed_s=0.01,
        )

    llm = FakeLLM([_step_finish("ok")])
    stack = build_agent(
        llm=llm,
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
        http_fetch_fn=fake_fetch,
    )
    assert "web_fetch" in stack.registry
    assert set(stack.registry.names()) == {"calculator", "search_docs", "web_fetch"}


def test_web_fetch_absent_without_fn():
    llm = FakeLLM([_step_finish("ok")])
    stack = build_agent(
        llm=llm,
        embedder=FakeEmbedder(dim=32),
        store=MemoryStore(),
    )
    assert "web_fetch" not in stack.registry
    assert set(stack.registry.names()) == {"calculator", "search_docs"}


# ---------- tool failure mid-multihop does not crash ----------

def test_multihop_recovers_from_tool_failure():
    script = [
        _step_tool("calculator", {"expression": "not valid"}),  # -> ValueError caught
        _step_tool("calculator", {"expression": "3*7"}),         # -> ok
        _step_finish("21", confidence="high"),
    ]
    stack, _ = _stack(script)
    out = stack.agent.run("q")

    assert out.tool_calls[0].ok is False
    assert out.tool_calls[1].ok is True
    assert out.final_answer == "21"
