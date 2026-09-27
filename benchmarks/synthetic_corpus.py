"""Deterministic synthetic corpus for scale benchmarks.

Purpose: measure the pipeline at a size where fixed costs stop dominating.
The 12-doc corpus in `evals/data/` is right for correctness; it is far too
small to say anything about throughput or p95.

Design:
- Deterministic: same `seed` -> byte-identical output. Committed results
  can be compared across runs.
- Realistic-ish: 3-5 sentences per doc, drawn from templates, with a
  unique signature token per doc so retrieval can be made meaningful.
- Configurable: `n_docs`, `seed`, `template_version`. Bumping the
  template version invalidates committed baselines on purpose.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

# Template version: bump to invalidate cached baselines.
TEMPLATE_VERSION = "v1"

_SENTENCE_TEMPLATES = [
    "The system processes {subject} using {method}.",
    "{subject} is a key component of the {domain} pipeline.",
    "We measured {subject} and observed a {metric} of {value}.",
    "In the {domain} context, {subject} interacts with {other}.",
    "The {method} approach to {subject} was validated on {dataset}.",
]

_SUBJECTS = [
    "the retriever", "the embedder", "the chunker", "the reranker",
    "the agent loop", "the vector store", "the dedup window",
    "the streaming ingestor", "the parallel consumer", "the metrics layer",
]

_METHODS = [
    "recursive splitting", "cosine similarity", "cross-encoder scoring",
    "token-bucket rate limiting", "hierarchical planning",
    "sliding-window dedup", "batched upsert", "round-robin prefetch",
]

_DOMAINS = [
    "retrieval", "ingestion", "orchestration", "observability",
    "evaluation", "streaming", "agentic", "multi-tenant",
]

_METRICS = [
    "p50 latency", "p95 latency", "throughput", "recall", "precision",
    "MRR", "hit-rate", "cost per query",
]

_DATASETS = [
    "an internal corpus", "a synthetic split", "a public benchmark",
    "a redacted sample", "a holdout set",
]


@dataclass(frozen=True)
class SyntheticDoc:
    id: str
    text: str
    source: str

    def as_row(self) -> dict:
        return {"id": self.id, "text": self.text, "source": self.source}


def _sentences(rng: random.Random, n: int) -> list[str]:
    out: list[str] = []
    for _ in range(n):
        tpl = rng.choice(_SENTENCE_TEMPLATES)
        out.append(
            tpl.format(
                subject=rng.choice(_SUBJECTS),
                method=rng.choice(_METHODS),
                domain=rng.choice(_DOMAINS),
                metric=rng.choice(_METRICS),
                value=f"{rng.uniform(0.1, 500.0):.2f}",
                other=rng.choice(_SUBJECTS),
                dataset=rng.choice(_DATASETS),
            )
        )
    return out


def generate_corpus(
    n_docs: int,
    *,
    seed: int = 42,
) -> list[SyntheticDoc]:
    """Return `n_docs` deterministic docs.

    Each doc contains a unique signature token of the form
    `sig-<zero-padded-index>`. A query for that token unambiguously
    targets one doc, which is what makes retrieval metrics meaningful
    even on a synthetic corpus.
    """
    if n_docs < 1:
        raise ValueError("n_docs must be >= 1")
    rng = random.Random(seed)

    docs: list[SyntheticDoc] = []
    for i in range(n_docs):
        idx = f"{i:04d}"
        doc_id = f"syn-{idx}"
        # Three unique tokens, not one: a single token is too easy to
        # confuse with common vocabulary under a bag-of-tokens embedder.
        # This makes the retrieval query discriminative.
        signature = f"sig-{idx} tag-{idx} key-{idx}"
        body = " ".join(_sentences(rng, rng.randint(3, 5)))
        # Signature must be a standalone whitespace token so the offline
        # FakeEmbedder (bag-of-tokens) can ground retrieval on it. A
        # bracketed form like "[sig-0000]" tokenizes differently from the
        # query "sig-0000" and retrieval never matches.
        text = f"{signature} {body}"
        docs.append(
            SyntheticDoc(id=doc_id, text=text, source=f"{doc_id}.txt")
        )
    return docs


def generate_queries(
    docs: list[SyntheticDoc],
    *,
    n_queries: int,
    seed: int = 43,
) -> list[tuple[str, str]]:
    """Return `n_queries` (question, expected_doc_id) pairs.

    Queries are just the signature token — deterministic and unambiguous.
    """
    if n_queries < 1:
        raise ValueError("n_queries must be >= 1")
    rng = random.Random(seed)
    picked = rng.sample(docs, k=min(n_queries, len(docs)))
    out: list[tuple[str, str]] = []
    for d in picked:
        idx = d.id.split("-")[1]  # "syn-0000" -> "0000"
        out.append((f"sig-{idx} tag-{idx} key-{idx}", d.id))
    return out
