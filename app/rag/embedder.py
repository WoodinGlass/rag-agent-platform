"""Provider-agnostic embedder interface.

`FakeEmbedder` is deterministic (hash bag-of-tokens) and requires no network,
which keeps unit tests fast and offline. Swap to `OpenAIEmbedder` by setting
EMBEDDER_PROVIDER=openai.
"""
from __future__ import annotations

import hashlib
import math
from abc import ABC, abstractmethod
from typing import Sequence


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_one(self, text: str) -> list[float]:
        return self.embed([text])[0]


class FakeEmbedder(Embedder):
    """Deterministic bag-of-tokens hashing embedder. L2-normalized."""

    def __init__(self, dim: int = 128):
        self.dim = dim

    def _vec(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        tokens = text.lower().split()
        if not tokens:
            return vec
        for tok in tokens:
            h = int.from_bytes(hashlib.sha256(tok.encode()).digest()[:8], "big")
            idx = h % self.dim
            sign = 1.0 if (h >> 8) & 1 else -1.0
            vec[idx] += sign
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]


class OpenAIEmbedder(Embedder):
    """Lazy import; only requires `openai` if actually instantiated."""

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ):
        try:
            from openai import OpenAI
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "openai not installed; run: pip install 'rag-agent-platform[llm]'"
            ) from e
        import os
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.dim = 1536

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=list(texts))
        return [d.embedding for d in resp.data]


def get_embedder(provider: str = "fake", **kwargs) -> Embedder:
    if provider == "fake":
        return FakeEmbedder(**kwargs)
    if provider == "openai":
        return OpenAIEmbedder(**kwargs)
    raise ValueError(f"unknown embedder provider: {provider}")
