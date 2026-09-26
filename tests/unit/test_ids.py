from app.core.ids import chunk_id, content_hash, doc_id, idempotency_key


def test_content_hash_stable():
    assert content_hash(b"hello") == content_hash(b"hello")
    assert content_hash(b"hello") != content_hash(b"world")


def test_doc_id_depends_on_version():
    h = content_hash(b"abc")
    assert doc_id(h, "v1") != doc_id(h, "v2")


def test_doc_id_stable():
    h = content_hash(b"abc")
    assert doc_id(h, "v1") == doc_id(h, "v1")


def test_chunk_id_padded():
    assert chunk_id("doc123", 7) == "doc123:0007"


def test_idempotency_key_order_matters():
    assert idempotency_key("a", "b") != idempotency_key("b", "a")
