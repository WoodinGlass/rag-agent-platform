# Architecture

Short tour of the system. For behaviour under failure, see `docs/runbooks/`.

## Layers
[client] --HTTP--> [FastAPI] --calls--> [Agent loop] --calls--> [Tools]
| | |
| | +-- search_docs -> RagPipeline.retrieve
| | +-- calculator (pure)
| | +-- web_fetch -> httpx adapter
| |
| +-- LLM (FakeLLM | OpenAI | Anthropic)
|
+-- middleware: correlation id, access log
+-- /metrics (in-process counters + histograms)

text

## Modules

| Module | Responsibility | Purity |
|---|---|---|
| `app/core/config.py` | Settings from env (pydantic-settings) | pure |
| `app/core/ids.py` | Deterministic ids + idempotency keys | pure |
| `app/core/logging.py` | JSON logs + correlation id contextvar | IO (stdout) |
| `app/core/metrics.py` | Counters + latency histograms | thread-safe state |
| `app/rag/chunker.py` | Recursive, version-tagged, deterministic | pure |
| `app/rag/embedder.py` | `Embedder` ABC, `FakeEmbedder`, lazy `OpenAIEmbedder` | IO in impls |
| `app/rag/store.py` | `VectorStore` ABC, `MemoryStore`, lazy `ChromaStore` | IO in impls |
| `app/rag/pipeline.py` | Idempotent ingest + retrieve | orchestrates |
| `app/tools/base.py` | `Tool` protocol (name + schema + callable) | pure |
| `app/tools/*.py` | Tool logic (search_docs, calculator, web_fetch) | pure |
| `app/tools/adapters/*.py` | IO boundaries injected into tools | IO |
| `app/agents/registry.py` | Register/resolve tools by name | pure |
| `app/agents/llm.py` | `LLM` protocol, `AgentStep` schema, `FakeLLM` | pure (FakeLLM) |
| `app/agents/prompts.py` | Versioned prompts | pure |
| `app/agents/schema.py` | `AgentOutput` (structured output contract) | pure |
| `app/agents/agent.py` | State machine: plan -> tool -> observe -> finish | orchestrates |
| `app/agents/bootstrap.py` | Factory: wire pipeline + tools + agent | pure assembly |
| `app/api/*` | HTTP surface: schemas, deps, middleware, routes | IO |
| `app/main.py` | FastAPI app, lifespan, error handler | IO |

## Key invariants

1. **Determinism where it matters.**
   Ingestion and retrieval are deterministic (hash + version). Generation
   is not — the schema (not the text) is the contract.

2. **Idempotent ingest.**
   `sha256(content) + chunker_version` -> stable `doc_id`. Re-ingest is a
   no-op. Bump `chunker_version` to re-chunk without collisions.

3. **Pure vs IO.**
   Tools are pure. IO lives in `adapters/` and is injected. Unit tests
   never touch network or disk.

4. **Boundary-guarded tool calls.**
   A tool that raises becomes `ToolCall(ok=False)`; the agent loop feeds
   the error back to the LLM and continues. No crash.

5. **Structured output enforced.**
   `AgentOutput` has `extra="forbid"`. Invalid LLM output hard-fails at
   the boundary with a correlation id, never silently.

6. **Correlation id everywhere.**
   Every request carries `X-Request-ID`. Response echoes it; every log
   line in-request carries it. Debugging = grep one cid.

## Failure-mode map

| Failure | Detected by | Runbook |
|---|---|---|
| LLM timeout / provider down | `request.error` + latency histogram | `runbooks/llm-timeout.md` |
| Vector store unreachable | healthz `degraded`, ingest errors | `runbooks/vector-store-down.md` |
| Schema drift (client or LLM) | `422` or `agent.invalid_step` | `runbooks/schema-violation.md` |

## Extension points

- **New tool:** implement in `app/tools/`, register in
  `app/agents/bootstrap.py`. No agent changes needed.
- **New LLM provider:** implement `LLM` protocol in `app/agents/llm.py`,
  select in `app/main.py` from `Settings.llm_provider`.
- **New vector backend:** implement `VectorStore` in `app/rag/store.py`,
  select in `app/main.py` from `Settings.vector_backend`.
- **New route:** add to `app/api/routes.py`, use `Annotated[..., Depends(...)]`
  aliases from `app/api/deps.py`.

All extension points follow the same pattern: **registry or ABC, chosen
by config**. No `if name == "x"` chains.
Cell 6 — Update README (link ke docs)
Tambahkan baris di section "Failure modes & runbooks (M4)":

python
%cd /content/rag-agent-platform

readme = open("README.md").read()

old = """## Failure modes & runbooks (M4)

See `docs/runbooks/`:

1. `llm-timeout.md` — retry with jitter, fallback model, cache hit
2. `vector-store-down.md` — circuit breaker, degraded retrieval
3. `schema-violation.md` — hard fail, correlation ID, sample payload"""

new = """## Failure modes & runbooks

Three failure modes are documented end-to-end (symptom -> triage ->
mitigation -> prevention -> signals):

1. [`llm-timeout.md`](docs/runbooks/llm-timeout.md) — provider slow or down;
   retry with jitter, fallback to `fake`, cache hit
2. [`vector-store-down.md`](docs/runbooks/vector-store-down.md) — store
   unreachable; restart, snapshot, or fall to `memory` (degraded)
3. [`schema-violation.md`](docs/runbooks/schema-violation.md) — client
   contract or LLM output violates schema; hard fail with cid

Architecture overview: [`docs/architecture.md`](docs/architecture.md)."""

if old in readme:
    readme = readme.replace(old, new)
    open("README.md", "w").write(readme)
    print("README updated")
else:
    print("pattern not found — manual edit")
