# Cross-encoder reranker benchmark

- Generated: `2026-09-27T02:01:38+00:00`
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- k=5 | chunk_size=200 | fetch multiplier=2 | warmup 2

## Quality

| Metric | Identity | Cross-encoder | Delta |
|---|---|---|---|
| hit_rate@5 | 1.0 | 0.9333 | -0.0667 |
| MRR@5 | 0.7189 | 0.88 | +0.1611 |
| precision@5 | 0.2533 | 0.2667 | +0.0134 |
| recall@5 | 0.9667 | 0.9 | -0.0667 |

## Latency (per query, ms)

| Metric | Identity | Cross-encoder | Delta |
|---|---|---|---|
| mean | 2.04 | 298.953 | +296.913 |
| p50 | 2.044 | 274.531 | +272.487 |
| p95 | 2.074 | 428.11 | +426.036 |
