# rag-agent-platform

> A production-minded RAG + AI agent backend with tool-calling, structured outputs, and a reproducible evaluation loop.

[![CI](https://github.com/<username>/rag-agent-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/<username>/rag-agent-platform/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11+-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Problem (one sentence)

Developers need a **reusable, testable, and observable RAG + agent backend** that ingests arbitrary documents, retrieves grounded context, executes tools, and returns **structured JSON** — instead of one-off notebooks glued to a single LLM provider.

## What this is / is not

| This is | This is not |
|---|---|
| Backend service (FastAPI) + evaluation harness | A UI / chatbot frontend |
| Provider-agnostic LLM layer (OpenAI / Anthropic) | Tied to a single vendor SDK |
| Deterministic ingestion + retrieval, non-deterministic generation | Fully deterministic pipeline |
| Portfolio-grade with CI, tests, Docker, evals | Production SaaS with billing / multi-tenant |

---

## Tech stack

- **Runtime:** Python 3.11+, FastAPI, Uvicorn
- **Orchestration:** LangChain / LangGraph
- **LLM providers:** OpenAI, Anthropic (pluggable)
- **Vector store:** Chroma (local dev) → Qdrant (prod)
- **Schemas:** Pydantic v2
- **Eval:** Ragas (retrieval + faithfulness), pytest markers
- **Observability:** structured JSON logs + correlation ID, metrics endpoint
- **Packaging:** `pyproject.toml` (single source of truth), `uv` / `pip`
- **Infra:** Docker + docker-compose, GitHub Actions CI

---

## Architecture

```text
 [docs] --> [Ingest] --> chunk --> embed --> [Vector Store]
                                                    |
                                                    v
 [query] --> [FastAPI] --> [RAG Pipeline] --> [Agent + Tools]
                                                    |
                                                    v
                                        [Structured JSON output]
```

### Design decisions (Phase 0 pre-flight)

| Decision | Choice | Rationale |
|---|---|---|
| **Deterministic vs non-deterministic** | Ingest + retrieval deterministic (hash-based idempotency); generation non-deterministic (UUID + dedup window) | Enables re-ingest without duplicates |
| **Payload vs pipeline** | Payload = documents + embeddings; Pipeline = chunk → embed → retrieve → generate | Clear separation for testing |
| **Storage tier** | Small structured: SQLite (dev) / Postgres (prod). Unstructured: local FS → S3. Vectors: Chroma → Qdrant | No refactor when scaling |
| **Ingestion mode** | Batch first (simple), streaming later behind flag | MVP velocity |
| **Idempotency key** | `sha256(file_bytes) + chunker_version` | Re-ingest is a no-op |
| **Failure modes** | LLM timeout → retry w/ jitter + fallback model. Vector store down → circuit breaker, cached retrieval. API down → 503 + correlation ID in logs | See `docs/runbooks/` |

---

## Repository layout

```text
rag-agent-platform/
├── app/
│   ├── agents/          # agent graph, tool registry, prompts
│   ├── rag/             # chunker, embedder, retriever, reranker
│   ├── tools/           # tool implementations (pure + IO separated)
│   ├── api/             # FastAPI routers, schemas, deps
│   ├── core/            # config, logging, ids, errors
│   └── main.py
├── evals/               # Ragas datasets + reports
├── tests/
│   ├── unit/            # fast, no IO (default)
│   └── integration/     # needs Docker / DB (marker: integration)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── docs/
│   ├── architecture.md
│   └── runbooks/        # 3 most common failures
├── scripts/             # one-off utilities (ingest_demo.py, etc.)
├── .github/workflows/   # ci.yml, eval.yml
├── .env.example
├── .gitignore
├── CHANGELOG.md
├── LICENSE
├── pyproject.toml
└── README.md
```

---

## Quickstart (local)

```bash
# 1. install
pip install -e ".[dev]"

# 2. config
cp .env.example .env
# edit .env: OPENAI_API_KEY=...

# 3. ingest demo corpus
python scripts/ingest_demo.py

# 4. run API
uvicorn app.main:app --reload

# 5. query
curl -X POST localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?"}'
```

## Quickstart (Docker)

```bash
docker compose -f docker/docker-compose.yml up --build
```

## Quickstart (Google Colab)

See `notebooks/00_setup_colab.ipynb` — mounts repo, sets secrets, runs smoke test.

---

## Testing

```bash
pytest -m "not integration"   # fast, no external deps
pytest -m integration         # spins up Docker (testcontainers)
```

## Evaluation

```bash
python -m evals.run --dataset evals/data/demo.jsonl --out evals/reports/
```

Latest report: `evals/reports/latest.md`

| Metric | Target | Latest |
|---|---|---|
| Faithfulness | ≥ 0.85 | – |
| Answer relevancy | ≥ 0.80 | – |
| Context precision | ≥ 0.75 | – |
| p95 latency (query) | ≤ 2.5 s | – |
| Cost / 1k queries | ≤ $0.50 | – |

---

## Failure modes & runbooks

See `docs/runbooks/`:

1. `llm-timeout.md` — retry with jitter, fallback model, cache hit
2. `vector-store-down.md` — circuit breaker, degraded retrieval
3. `schema-violation.md` — hard fail, correlation ID, sample payload

---

## Contributing

Conventional commits. One PR = one milestone checkbox. `pre-commit` runs ruff + mypy.

---

## Roadmap (production-scale)

Each milestone ships **runnable, tested, and documented** code — not stubs.

### M1 — Local RAG (deterministic core)

- [ ] `app/rag/`: chunker (recursive + version), embedder (provider-agnostic), Chroma store
- [ ] Idempotent ingest: `sha256(file) + chunker_version` as key
- [ ] `scripts/ingest_demo.py` ingests a sample corpus
- [ ] Retrieval endpoint returns top-k with scores + source spans
- [ ] Unit tests (chunker edge cases, embedder mock)
- [ ] **Exit criteria:** `pytest -m "not integration"` green, retrieval hit-rate@5 ≥ 0.8 on demo set

### M2 — Agent + tools

- [ ] LangGraph agent with tool registry (`register(name, fn)`)
- [ ] 3 tools: `search_docs`, `calculator`, `web_fetch` (timeout + retry)
- [ ] Structured JSON output enforced via Pydantic schema
- [ ] Pure logic vs IO separated (tools pure, adapters in `tools/adapters/`)
- [ ] **Exit criteria:** agent answers multi-hop question in demo notebook, schema validation passes

### M3 — API + Docker

- [ ] FastAPI routes: `/ingest`, `/query`, `/healthz`, `/metrics`
- [ ] Correlation ID middleware, structured JSON logs
- [ ] `docker/Dockerfile` (multi-stage), `docker-compose.yml` (API + Qdrant)
- [ ] `.env.example` fully documents every knob
- [ ] Config via `pydantic-settings`, no hardcoded strings
- [ ] **Exit criteria:** `docker compose up` → `curl /query` returns valid JSON

### M4 — CI + eval report

- [ ] GitHub Actions: lint (ruff) → type (mypy) → unit test → build → smoke
- [ ] Integration tests behind `integration` marker (testcontainers)
- [ ] Ragas eval job: faithfulness, answer relevancy, context precision
- [ ] `evals/reports/` committed as artifacts; README badge shows latest
- [ ] `docs/runbooks/` for 3 top failures (LLM timeout, store down, bad schema)
- [ ] **Exit criteria:** CI green on `main`, eval report published, changelog updated

### M5+ (backlog — post-MVP)

- [ ] Streaming ingestion (Kafka / S3 events)
- [ ] Reranker (cross-encoder) behind flag
- [ ] Multi-tenant isolation + auth
- [ ] Cost model + benchmarks (`docs/benchmarks.md`)
- [ ] OpenTelemetry traces

---

## License

MIT — see [`LICENSE`](LICENSE).
