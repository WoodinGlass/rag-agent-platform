import pytest

from app.rag.chunker import CHUNKER_VERSION, chunk_text


def test_short_text_one_chunk():
    chunks = chunk_text("hello world", "d1", size=100)
    assert len(chunks) == 1
    assert chunks[0].text == "hello world"
    assert chunks[0].start == 0
    assert chunks[0].end == 11


def test_paragraph_split_respects_size():
    text = "a" * 40 + "\n\n" + "b" * 40 + "\n\n" + "c" * 40
    chunks = chunk_text(text, "d1", size=50)
    assert len(chunks) >= 3
    assert all(len(c.text) <= 50 for c in chunks)


def test_deterministic():
    text = "para one.\n\npara two.\n\npara three."
    a = chunk_text(text, "d1", size=20)
    b = chunk_text(text, "d1", size=20)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert [c.text for c in a] == [c.text for c in b]


def test_version_tag_propagates():
    chunks = chunk_text("hello", "d1", size=100, version="v9")
    assert all(c.chunker_version == "v9" for c in chunks)


def test_default_version():
    chunks = chunk_text("hello", "d1", size=100)
    assert chunks[0].chunker_version == CHUNKER_VERSION


def test_empty_text_no_chunks():
    assert chunk_text("", "d1") == []
    assert chunk_text("   ", "d1") == []


def test_invalid_size():
    with pytest.raises(ValueError):
        chunk_text("x", "d1", size=0)


def test_chunk_ids_unique_and_ordered():
    text = "\n\n".join(f"para {i}" for i in range(20))
    chunks = chunk_text(text, "d1", size=30)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for i, c in enumerate(chunks):
        assert c.chunk_id.endswith(f"{i:04d}")


def test_offsets_monotonic():
    text = "aaa.\n\nbbb.\n\nccc."
    chunks = chunk_text(text, "d1", size=8)
    starts = [c.start for c in chunks]
    assert starts == sorted(starts)
