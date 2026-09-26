# Benchmarks

Reproducible measurements of the **offline** pipeline. Two kinds of
numbers live here, always labelled:

- **Measured** — actual wall-clock on a specific machine, using
  `FakeEmbedder` + `MemoryStore` + `FakeLLM` (no network, no API cost).
- **Projected** — cost model derived from published list prices and
  token-count estimates. Not a claim of production performance.

Regenerate with:

```bash
python -m benchmarks.run --n 100 --k 5
Full report: benchmarks/results/latest.md ·
Raw JSON: benchmarks/results/latest.json.

Setup
Knob    Value
Corpus  12 documents (evals/data/corpus.jsonl)
Eval set    15 questions (evals/data/eval_set.jsonl)
Chunk size  200
Embedder    FakeEmbedder(dim=256) — deterministic, offline
Vector store    MemoryStore (in-process)
Warmup  10 queries (excluded from stats)
Measured queries    100
Ingest (measured)
Ingestion = chunk → embed → store, per document, idempotent on
sha256(content) + chunker_version.

Metric  Value
Total (12 docs) 3.469 ms
Per document    0.289 ms
This measures the local pipeline. It does not include the cost of
fetching bytes from an object store (network-bound; out of scope here).

Retrieve latency (measured)
pipeline.retrieve(question, k=5) — embed query + cosine query + sort.

Metric  ms
mean    2.188
p50 2.067
p90 2.257
p95 3.168
p99 4.154
min 1.834
max 4.180
At MemoryStore scale (12 chunks) retrieval is dominated by embedding
the query, not by the vector search. The interesting question is what
happens at 10k+ chunks with a real ANN index (Chroma / Qdrant). That's
on the M5 backlog; this baseline tells us the pipeline overhead is
~sub-millisecond, so the ceiling is the vector store, not our code.

Cost model (projected)
Model parameters, from the report:

Knob    Value   Source
Embedding   $0.02 / 1M tokens   text-embedding-3-small list price
LLM input   $0.15 / 1M tokens   gpt-4o-mini list price
LLM output  $0.60 / 1M tokens   gpt-4o-mini list price
Context tokens / query  200 assumption (top-5 chunks)
Output tokens / query   80  assumption (short answer)
Tokenizer   whitespace  estimate, not tiktoken
Results:

Item    Value
Corpus embedding (one-time) $0.0000070 (369 tokens)
Per query (input + context + output)    $0.0000790
Per 1,000 queries   $0.0790
Cost is dominated by LLM output tokens. Levers in a real deployment:
shorter answers, caching for repeated questions, smaller models for
easy questions.

Honest gaps
No real provider numbers yet. These run offline. Add
--embedder openai with OPENAI_API_KEY to get real provider
latency; the script already supports it.

No concurrency. Single-threaded. Real p95 under load is a
different number — that's the job of a load-test milestone
(not scheduled).

No ANN at scale. See retrieve note above.

Cost model is list-price only. Volume discounts, prompt caching,
and batch APIs are not modelled.

Whitespace tokenizer. Real token counts (tiktoken / anthropic)
will differ by ±20%. Treat cost figures as order-of-magnitude.
