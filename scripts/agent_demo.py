"""End-to-end agent demo: multi-hop question with 3 tools wired.

Runs offline with a scripted FakeLLM by default (so it works in CI and
Colab with no API key). The scripted replies simulate what a well-behaved
LLM would do — search, then calculate, then answer with a citation.

Usage:
    python scripts/agent_demo.py
    python scripts/agent_demo.py --question "..."
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.agents.bootstrap import build_agent
from app.agents.llm import FakeLLM
from app.core.logging import configure_logging
from app.rag.embedder import FakeEmbedder
from app.rag.store import MemoryStore

DEFAULT_QUESTION = "How many letters are in the word 'cat', multiplied by 7?"

CORPUS_DIR = Path("evals/data/demo_corpus")


def _load_corpus() -> list[tuple[str, bytes]]:
    """Use evals corpus files if present, else fall back to inline docs."""
    docs: list[tuple[str, bytes]] = []
    if CORPUS_DIR.exists():
        for p in sorted(CORPUS_DIR.glob("*.txt")):
            docs.append((p.name, p.read_bytes()))
    if docs:
        return docs
    return [
        ("cat.txt", b"The cat sat on the mat. The word cat has 3 letters."),
        ("dog.txt", b"Dogs bark loudly. The word dog has 3 letters."),
        ("quantum.txt", b"Quantum computing uses qubits and superposition."),
    ]


def _scripted_multihop() -> list[str]:
    """A minimal, deterministic multi-hop plan."""
    return [
        json.dumps(
            {
                "action": "tool",
                "tool": "search_docs",
                "args": {"query": "cat letters", "k": 2},
                "thought": "Find a source that states how many letters 'cat' has.",
            }
        ),
        json.dumps(
            {
                "action": "tool",
                "tool": "calculator",
                "args": {"expression": "3*7"},
                "thought": "Multiply the letter count by 7.",
            }
        ),
        json.dumps(
            {
                "action": "finish",
                "final_answer": "The word 'cat' has 3 letters; 3 x 7 = 21.",
                "citations": [
                    {
                        "doc_id": "d",
                        "chunk_id": "d:0000",
                        "quote": "cat has 3 letters",
                    }
                ],
                "confidence": "high",
                "refused": False,
                "reason": "",
            }
        ),
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--question", default=DEFAULT_QUESTION)
    ap.add_argument("--max-steps", type=int, default=6)
    args = ap.parse_args()

    configure_logging(level="WARNING", json_output=False)

    docs = _load_corpus()
    stack = build_agent(
        llm=FakeLLM(_scripted_multihop()),
        embedder=FakeEmbedder(dim=128),
        store=MemoryStore(),
        chunk_size=200,
        max_steps=args.max_steps,
    )

    for name, data in docs:
        r = stack.pipeline.ingest(data, source=name)
        print(f"ingest {name:<14} doc_id={r.doc_id[:8]} chunks={r.n_chunks}")

    print(f"\nquestion: {args.question}")
    out = stack.agent.run(args.question)

    print("\n--- agent output (JSON) ---")
    print(json.dumps(json.loads(out.to_json()), indent=2))

    print("\n--- tool trace ---")
    for i, tc in enumerate(out.tool_calls, 1):
        status = "ok" if tc.ok else f"fail: {tc.error}"
        print(f"  {i}. {tc.tool} ({status}, {tc.latency_ms} ms)")


if __name__ == "__main__":
    main()
