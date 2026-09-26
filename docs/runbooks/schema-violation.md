# Runbook: Schema Violation

## Symptom

- `/query` returns `500` with `detail` mentioning validation.
- Log line: `request.error` or `agent.invalid_step`.
- `/ingest` returns `422` for a valid-looking payload (client-side schema).
- `/query` returns `200` but the `output` shape is unexpected (contract drift).

## Blast radius

Schema violations come in three flavors. Identify which before acting.

| Flavor | Where caught | Status |
|---|---|---|
| **Request schema** — client sends bad JSON | FastAPI / Pydantic at HTTP boundary | `422` |
| **Agent step schema** — LLM emits invalid `AgentStep` JSON | `app/agents/agent.py` at `model_validate_json` | Fed back to LLM, retried |
| **Final output schema** — LLM emits invalid `AgentOutput` | `app/agents/schema.py` at `AgentOutput(...)` | `500` with `correlation_id` |

**Hard-fail policy:** `AgentOutput` has `extra="forbid"`. If the LLM
invents a field, we raise — we do **not** silently drop it.

## Triage

1. **Get the exact failure from logs.**
   ```bash
   docker logs rag-api | grep -E "schema|validation|invalid" | tail -30
   # or by cid
   docker logs rag-api | grep '"cid":"<CID>"' | tail -10
Reproduce request shape.

bash
# was it the client's fault or the LLM's?
curl -sS -X POST localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"...","extra_field":1}' | jq
# 422 with `{"detail":[...]}` -> client-side
If the LLM produced invalid output, replay with the same question
in a dev shell and print the raw step:

python
from app.agents.llm import FakeLLM  # or real provider
# ... run agent, capture raw LLM strings
Mitigation
A. Client sent extra/wrong fields (422)

Fix the client. The contract is:

json
// POST /query
{ "question": "string (1..4000, non-empty after strip)" }

// POST /ingest
{
  "content_base64": "base64 string (valid, non-empty)",
  "source": "string (<=512, optional)",
  "metadata": { "any": "json-serializable" }
}
FastAPI response will include the field name and reason. Nothing to do on
the server side.

B. LLM emitted invalid AgentStep

The agent already feeds the error back and retries (agent.invalid_step
log). If it retries repeatedly and exhausts max_steps:

Lower temperature (config); the loop already runs deterministic on
FakeLLM.

Tighten SYSTEM_PROMPT in app/agents/prompts.py — the schema is
already inline, ensure the model is told "no prose, JSON only".

Swap to a stronger model for one release.

C. LLM emitted invalid AgentOutput (500)

This is the hard-fail case. The correlation id in the error body links to
the exact raw LLM response in the log. Options:

Fix the model response — usually the answer is missing
final_answer when refused=false, or refused=true without reason.

Widen the schema intentionally (only if the change is defensible):
e.g. allow refused=true with empty reason if the caller handles it.
Do not relax extra="forbid" casually — that invites silent drift.

Add a repair step: on ValidationError, ask the LLM to reformat
the previous answer to the schema. This trades one extra call for
availability. (Roadmap M5+.)

Prevention
extra="forbid" everywhere — catches client and LLM typos early.

Schema is the contract — app/agents/schema.py and
app/api/schemas.py are the single source of truth. Any change is a
breaking change to be documented in CHANGELOG.md.

Tests — every schema has round-trip JSON tests
(tests/unit/test_agent_schema.py, tests/unit/test_api_schemas.py).

Repair loop (roadmap M5+): retry-once-then-refuse for LLM output
that fails validation.

OpenAPI export — /docs is the live contract; keep it in sync by
changing the Pydantic models, never the route signatures.

Signals
Signal  Where   Meaning
agent.invalid_step  app log LLM step failed validation, retry in progress
request.error with detail=...ValidationError... app log Hard-fail on final output
422 responses   access log  Client-side contract violation
/metrics query.refused  app metrics Refusals vs failures
Related
app/agents/schema.py — AgentOutput, Citation, ToolCall

app/agents/llm.py — AgentStep (intermediate LLM schema)

app/api/schemas.py — HTTP request/response schemas

app/agents/agent.py — _build_output (hard-fail path)
