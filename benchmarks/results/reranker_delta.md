# Reranker delta

- Generated: `2026-09-26T18:13:29+00:00`
- Corpus: 12 docs | k=5 | fetch multiplier=2
- Embedder: `fake` | warmup 10

## Quality (retrieval metrics)

| Metric | Identity (baseline) | Fake (reranked) | Delta |
|---|---|---|---|
| hit_rate@5 | 1.0 | 1.0 | +0.0000 |
| MRR@5 | 0.7189 | 0.83 | +0.1111 |
| precision@5 | 0.2533 | 0.2667 | +0.0134 |
| recall@5 | 0.9667 | 0.9667 | +0.0000 |

## Latency (per query, ms)

| Metric | Identity | Fake | Delta |
|---|---|---|---|
| mean | 3.657 | 2.752 | -0.905 |
| p50 | 3.939 | 2.055 | -1.884 |
| p95 | 4.008 | 4.786 | +0.778 |

> `FakeReranker` is a deterministic token-overlap heuristic, not a
> cross-encoder. This measures the *pipeline wiring cost* and gives
> a sanity check on ranking change, not a claim about cross-encoder
> quality. A real cross-encoder (`sentence-transformers`) can be
> plugged in via `CrossEncoderReranker` for production eval.
