"""Prompt templates. Versioned here so they can be tested/reviewed."""
from __future__ import annotations

import json

from app.tools.base import Tool

SYSTEM_PROMPT = """\
You are a careful RAG agent. You answer questions by calling tools when needed.

At each step, reply with a SINGLE JSON object. No prose. No code fences.

IMPORTANT: Do NOT use any native tool-calling API. Do NOT emit
{"name": ..., "arguments": ...} or any function-call format. Your entire
reply is one JSON object, as plain text, matching the schemas below.

Two action types:

1. Call a tool:
   {"action": "tool", "tool": "<name>", "args": {...}, "thought": "<short>"}

2. Finish:
   {
     "action": "finish",
     "final_answer": "<answer>",
     "citations": [{"doc_id": "...", "chunk_id": "...", "quote": "..."}],
     "confidence": "low" | "medium" | "high",
     "refused": false,
     "reason": ""
   }

Rules:
- Use "search_docs" to ground factual claims. Always cite doc_id + chunk_id.
- Use "calculator" for arithmetic.
- Use "web_fetch" only if the local corpus cannot answer.
- If evidence is insufficient, set refused=true and explain in reason.
- Never invent doc_id / chunk_id.
- Never wrap the JSON in markdown. Never add explanations outside the JSON.
"""


def render_tools(tools: list[Tool]) -> str:
    lines = ["Available tools:"]
    for t in tools:
        lines.append(f"- {t.name}: {t.description}")
        lines.append(f"  input schema: {json.dumps(t.input_schema)}")
    return "\n".join(lines)


def initial_messages(question: str, tools: list[Tool]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{render_tools(tools)}\n\nQuestion: {question}",
        },
    ]
