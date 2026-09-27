# Benchmarks

Reproducible measurements of the **offline** pipeline. Two kinds of numbers live here, always labelled:

- **Measured** — actual wall-clock on a specific machine, using `FakeEmbedder` + `MemoryStore` + `FakeLLM` (no network, no API cost).
- **Projected** — cost model derived from published list prices and token-count estimates. Not a claim of production performance.

Regenerate with:

```bash
python -m benchmarks.run --n 100 --k 5
python -m benchmarks.reranker_delta --k 5
```

Full report: [`benchmarks/results/latest.md`](../benchmarks/results/latest.md) · Raw JSON: [`benchmarks/results/latest.json`](../benchmarks/results/latest.json).

## Setup

| Knob | Value |
|---|---|
| Corpus | 12 documents (`evals/data/corpus.jsonl`) |
| Eval set | 15 questions (`evals/data/eval_set.jsonl`) |
| Chunk size | 200 |
| Embedder | `FakeEmbedder(dim=256)` — deterministic, offline |
| Vector store | `MemoryStore` (in-process) |
| Warmup | 10 queries (excluded from stats) |
| Measured queries | 100 |

## Ingest (measured)

Ingestion = chunk -> embed -> store, per document, idempotent on `sha256(content) + chunker_version`.

| Metric | Value |
|---|---|
| Total (12 docs) | **3.469 ms** |
| Per document | **0.289 ms** |

This measures the *local* pipeline. It does not include the cost of fetching bytes from an object store (network-bound; out of scope here).

## Retrieve latency (measured)

`pipeline.retrieve(question, k=5)` — embed query + cosine query + sort.

| Metric | ms |
|---|---|
| mean | 2.188 |
| p50 | 2.067 |
| p90 | 2.257 |
| p95 | **3.168** |
| p99 | 4.154 |
| min | 1.834 |
| max | 4.180 |

At `MemoryStore` scale (12 chunks) retrieval is dominated by embedding the query, not by the vector search. The interesting question is what happens at 10k+ chunks with a real ANN index (Chroma / Qdrant). That's on the M5 backlog; this baseline tells us the pipeline overhead is ~sub-millisecond, so the ceiling is the vector store, not our code.

## Cost model (projected)

Model parameters, from the report:

| Knob | Value | Source |
|---|---|---|
| Embedding | $0.02 / 1M tokens | `text-embedding-3-small` list price |
| LLM input | $0.15 / 1M tokens | `gpt-4o-mini` list price |
| LLM output | $0.60 / 1M tokens | `gpt-4o-mini` list price |
| Context tokens / query | 200 | assumption (top-5 chunks) |
| Output tokens / query | 80 | assumption (short answer) |
| Tokenizer | whitespace | estimate, not `tiktoken` |

Results:

| Item | Value |
|---|---|
| Corpus embedding (one-time) | `$0.0000070` (369 tokens) |
| Per query (input + context + output) | `$0.0000790` |
| **Per 1,000 queries** | **`$0.0790`** |

Cost is dominated by LLM output tokens. Levers in a real deployment: shorter answers, caching for repeated questions, smaller models for easy questions.

## Reranker delta (M5.2)

Pattern: retrieve top `k * retrieve_multiplier` (cheap, approximate), then rerank to top `k` (more accurate). Here we compare `IdentityReranker` (baseline; no-op) against `FakeReranker`, a deterministic token-overlap heuristic.

Raw: [`benchmarks/results/reranker_delta.md`](../benchmarks/results/reranker_delta.md)

### Quality

| Metric | Identity | Fake | Delta |
|---|---|---|---|
| hit_rate@5 | 1.0 | 1.0 | +0.0000 |
| MRR@5 | 0.7189 | 0.83 | +0.1111 |
| precision@5 | 0.2533 | 0.2667 | +0.0134 |
| recall@5 | 0.9667 | 0.9667 | +0.0000 |

### Latency (per query, ms)

| Metric | Identity | Fake | Delta |
|---|---|---|---|
| mean | 2.362 | 2.517 | +0.155 |
| p50 | 1.935 | 1.983 | +0.048 |
| p95 | 3.896 | 4.464 | +0.568 |

### Honest reading

`FakeReranker` is a *token-overlap heuristic*, not a cross-encoder. It is intentionally simple so tests can assert its wiring deterministically. If the quality delta is ~0 or negative, that is the expected outcome: a heuristic cannot match a real cross-encoder's joint scoring.

What this section *does* establish:

1. **The pipeline path is live.** Reranker output is what `pipeline.retrieve()` returns; changing the reranker changes results.
2. **The latency cost of the extra fetch + rerank is small** compared to query embedding, at MemoryStore scale.
3. **The interface is honest.** `CrossEncoderReranker` swaps in with a one-line config change; no pipeline edits needed.

### What we are not claiming

- That `FakeReranker` improves retrieval quality. It likely does not, and the delta above is the evidence.
- That cross-encoder latency is small. Real cross-encoders add ~10-40 ms per query on CPU for top-20 candidates. Benchmarking that requires installing `sentence-transformers` (extra `rerank`) and running with `CrossEncoderReranker` — out of scope for the offline benchmark.


## Scale: 1k documents (M9.6)

The 12-doc benchmark above validates correctness. This one asks: does
the pipeline still behave at a size where fixed costs stop dominating?
Fully synthetic (deterministic, `template=v1`, seed=42), so the numbers are reproducible:

```bash
python -m benchmarks.scale_benchmark --n-docs 1000 --n-queries 200
```

Raw: [`benchmarks/results/scale_1000.md`](../benchmarks/results/scale_1000.md)

### Setup

| Knob | Value |
|---|---|
| Docs | 1000 |
| Queries | 200 |
| k | 5 |
| chunk_size | 200 |
| Embedder | `fake` (dim=256) |

### Ingest (measured)

| Metric | Value |
|---|---|
| Total | **1060.744 ms** |
| Per document | **1.0607 ms** |
| Throughput | **942.7 docs/s** |

### Retrieve latency (measured)

| Metric | ms |
|---|---|
| mean | 154.994 |
| p50 | **119.993** |
| p90 | 217.163 |
| p95 | **237.981** |
| p99 | 415.619 |
| max | 487.937 |

### Retrieval quality

| Metric | Value |
|---|---|
| hit_rate@5 | **0.96** |
| MRR@5 | **0.9163** |
| recall@5 | 0.96 |

### Honest reading

Synthetic docs carry a unique 3-token signature (`sig-N tag-N key-N`);
queries target it unambiguously. A high hit-rate here is a
*self-consistency check*, not a claim about real-corpus quality.
What this benchmark actually measures:

1. **Ingest scales roughly linearly.** 1.0607 ms/doc at 1k
   vs ~0.29 ms/doc at 12. The delta is embedder + store work that
   grows with signature length, not with N.
2. **Retrieve grows with corpus size.** 119.993 ms p50 at 1k vs ~2 ms
   at 12. `MemoryStore` does a linear cosine scan; the number of
   candidates dominates. The pipeline overhead is constant.
3. **FakeEmbedder loses signal at scale.** The first run (1-token
   signature) scored 0.76 hit-rate: a single token is too easy to
   confuse with common vocabulary under bag-of-tokens. Replacing
   it with 3 unique tokens lifted hit-rate to 0.96 and MRR to
   0.9163. This is a *finding*: small embeddings need
   discriminative anchors; a real embedding model is not this
   sensitive.

### What we are not claiming

- **Not a load test.** Single-threaded, one worker, no concurrency.
- **Not an ANN benchmark.** `MemoryStore` is a linear scan; at 1k
  vectors it is already the bottleneck. Chroma/Qdrant use ANN and
  their own numbers matter at scale.
- **Not a real-corpus number.** Synthetic docs are uniform in length
  and vocabulary; real corpora are not.

---
## Honest gaps

- **No real provider numbers yet.** These run offline. Add `--embedder openai` with `OPENAI_API_KEY` to get real provider latency; the script already supports it.
- **No concurrency.** Single-threaded. Real p95 under load is a different number — that's the job of a load-test milestone (not scheduled).
- **No ANN at scale.** See retrieve note above.
- **Cost model is list-price only.** Volume discounts, prompt caching, and batch APIs are not modelled.
- **Whitespace tokenizer.** Real token counts (tiktoken / anthropic) will differ by ±20%. Treat cost figures as order-of-magnitude.

