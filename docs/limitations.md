# Known limitations

Every production-minded repository should say out loud what it does
**not** do yet. This is the honest list. Where possible, each item links
to where it would be fixed.

## Scope

- **Not multi-tenant by default.** Tenant auth is opt-in
  (`TENANT_AUTH_ENABLED`); when on, it is a simple `X-API-Key` map. No
  key rotation, no per-tenant rate limits, no audit log.
- **No authorization, only authentication.** Any valid key can ingest
  and query. There is no role separation (reader vs. writer vs. admin).
- **No PII handling.** Documents are ingested as-is. There is no
  redaction, classification, or retention policy.

## Correctness

- **"Effectively once", not exactly once.** Delivery is at-least-once;
  correctness relies on idempotent ingestion plus a bounded dedup
  window. Rationale and failure modes:
  [`exactly-once.md`](exactly-once.md).
- **Dedup state is bounded.** The in-memory LRU (default) loses entries
  when it fills; the SQLite adapter persists but is not pruned
  automatically unless you call `prune()`. Durable + auto-pruned dedup
  would need a small KV store or a compacted topic.
- **Reranker is CPU-bound.** The cross-encoder adds ~270 ms p50 on CPU
  for `k * multiplier = 10` candidates. We documented the trade-off
  (see [`benchmarks.md`](benchmarks.md) § Real cross-encoder) but did
  not implement batching, GPU offload, or a smaller model as defaults.
- **Chunker is a heuristic.** Recursive paragraph/sentence splitting
  works well on prose. It does not understand tables, code blocks, or
  PDF layout. No semantic chunking.

## Reliability

- **No circuit breaker.** `docs/runbooks/llm-timeout.md` describes the
  intended behaviour; today the retry-with-jitter exists, the
  circuit-breaker part is a documented follow-up.
- **No rate limiting.** `/ingest` and `/query` accept requests without
  a request-size cap or per-tenant throttle.
- **No concurrency test.** Benchmarks measure single-threaded latency.
  p95 under load is a different number and is not measured here.
- **Docker smoke test is shallow.** The CI `docker` job validates that
  the image builds, boots, and answers `/healthz`. It does not exercise
  `/query` end-to-end against the container.

## Observability

- **Metrics are in-process JSON.** `/metrics` returns a JSON snapshot,
  not a Prometheus exposition format. Point a scraper at it only after
  adding an adapter.
- **Tracing has no persistent backend by default.** The `console`
  exporter writes spans to stdout; the OTLP exporter ships to whatever
  collector you point it at, but no collector config ships except the
  dev `logging` one.
- **No tracing across process boundaries.** `X-Request-ID` correlates
  HTTP requests. Nothing propagates that id into the streaming worker's
  partitions.

## Data

- **No schema migrations.** `SQLiteDedupStore` creates its table
  idempotently; there is no migration framework. Changing the schema in
  a future version would require manual migration.
- **Chroma is the only persistent vector adapter in-tree.** Qdrant is
  in `docker-compose.yml` and in the design-decisions table, but
  `QdrantStore` is not implemented.
- **No S3 state persistence.** The S3 event source tracks emitted and
  committed keys in memory. A worker restart re-lists the bucket; the
  ingestor's dedup absorbs the redelivery. State persistence is a
  documented follow-up.

## Testing

- **Benchmark corpus is tiny (12 documents).** Enough to validate the
  pipeline and catch regressions; not enough to claim scale. A 1k+
  document run is on the M9+ backlog.
- **Coverage floor is 85%, not 90%+.** A few files
  (`app/core/logging.py`, `app/rag/chunker.py`) are below the average;
  the total clears the threshold because other modules are 100%.
- **Provider smoke test is a smoke test.** Six tests, one pass. Not a
  load test, not a quality eval.

## Not planned (deliberate)

- **Billing / multi-tenant SaaS features.** Out of scope; this is a
  backend, not a product.
- **UI.** The API and OpenAPI docs are the interface.
- **Fine-tuning.** Provider-agnostic prompting is the design; model
  training is not.

## How to read this list

If an item matters to your use case, the honest answer is: "the design
allows it, the implementation does not ship it yet, and the runbook or
docs explain the trade-off." That is the standard this repository
holds itself to.
