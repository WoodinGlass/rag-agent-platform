import math

from app.rag.embedder import FakeEmbedder, get_embedder


def _cos(x, y):
    return sum(a * b for a, b in zip(x, y))


def test_fake_embedder_deterministic():
    e = FakeEmbedder(dim=32)
    assert e.embed(["hello world"])[0] == e.embed(["hello world"])[0]


def test_fake_embedder_dim():
    assert len(FakeEmbedder(dim=32).embed_one("hello")) == 32


def test_fake_embedder_normalized():
    v = FakeEmbedder(dim=64).embed_one("the quick brown fox")
    assert abs(math.sqrt(sum(x * x for x in v)) - 1.0) < 1e-6


def test_similar_texts_more_similar():
    e = FakeEmbedder(dim=128)
    q = e.embed_one("retrieval augmented generation")
    a = e.embed_one("retrieval augmented generation is useful")
    b = e.embed_one("banana smoothie recipe")
    assert _cos(q, a) > _cos(q, b)


def test_get_embedder_factory():
    assert isinstance(get_embedder("fake"), FakeEmbedder)
