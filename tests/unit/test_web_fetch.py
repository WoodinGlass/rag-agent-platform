import pytest

from app.tools.adapters.http import HttpResponse, make_http_fetch
from app.tools.web_fetch import (
    DEFAULT_TEXT_LIMIT,
    format_response,
    make_web_fetch_tool,
)


def _resp(
    text: str = "hello",
    status: int = 200,
    url: str = "https://x.test/",
) -> HttpResponse:
    return HttpResponse(
        url=url,
        status=status,
        text=text,
        headers={"content-type": "text/plain"},
        elapsed_s=0.123,
    )


# ---------- format_response (pure) ----------

def test_format_shape():
    out = format_response(_resp(), 100)
    assert out["url"] == "https://x.test/"
    assert out["status"] == 200
    assert out["ok"] is True
    assert out["text"] == "hello"
    assert out["truncated"] is False
    assert out["content_type"] == "text/plain"


def test_format_truncates():
    out = format_response(_resp(text="x" * 5000), 100)
    assert len(out["text"]) == 100
    assert out["truncated"] is True


def test_format_not_ok_for_4xx():
    assert format_response(_resp(status=404), 100)["ok"] is False


def test_format_elapsed_rounded():
    out = format_response(_resp(), 100)
    assert isinstance(out["elapsed_s"], float)


# ---------- tool wrapper ----------

def test_tool_calls_fetch_fn():
    calls: list[tuple[str, int]] = []

    def fake(url: str, max_bytes: int):
        calls.append((url, max_bytes))
        return _resp()

    t = make_web_fetch_tool(fake)
    out = t.call(url="https://a.test/p", max_chars=50)
    assert out["text"] == "hello"
    assert len(calls) == 1
    assert calls[0][0] == "https://a.test/p"


def test_tool_default_max_chars():
    t = make_web_fetch_tool(lambda u, b: _resp(text="x" * 10000))
    out = t.call(url="https://a.test/")
    assert len(out["text"]) == DEFAULT_TEXT_LIMIT


def test_tool_clamps_max_chars_above():
    t = make_web_fetch_tool(lambda u, b: _resp(text="x" * 10000))
    out = t.call(url="https://a.test/", max_chars=999999)
    assert len(out["text"]) == DEFAULT_TEXT_LIMIT


def test_tool_clamps_max_chars_below():
    t = make_web_fetch_tool(lambda u, b: _resp(text="abc"))
    out = t.call(url="https://a.test/", max_chars=0)
    assert len(out["text"]) == 1


# ---------- url validation ----------

def test_reject_empty_url():
    t = make_web_fetch_tool(lambda u, b: _resp())
    with pytest.raises(ValueError, match="non-empty"):
        t.call(url="")


def test_reject_non_http_scheme():
    t = make_web_fetch_tool(lambda u, b: _resp())
    with pytest.raises(ValueError, match="http/https"):
        t.call(url="ftp://x.test/")


def test_reject_no_host():
    t = make_web_fetch_tool(lambda u, b: _resp())
    with pytest.raises(ValueError, match="host"):
        t.call(url="http:///path")


def test_reject_non_string_url():
    t = make_web_fetch_tool(lambda u, b: _resp())
    with pytest.raises(ValueError):
        t.call(url=123)


# ---------- tool metadata ----------

def test_spec_shape():
    t = make_web_fetch_tool(lambda u, b: _resp())
    assert t.name == "web_fetch"
    assert t.spec()["parameters"]["required"] == ["url"]
    assert "http" in t.tags


# ---------- real adapter with pytest-httpx (offline mock) ----------

def test_http_adapter_success(httpx_mock):
    httpx_mock.add_response(
        url="https://api.test/hello", text="world", status_code=200
    )
    fetch = make_http_fetch(timeout_s=2, max_retries=0)
    r = fetch("https://api.test/hello")
    assert r.status == 200
    assert r.text == "world"
    assert r.ok()


def test_http_adapter_retry_then_success(httpx_mock):
    httpx_mock.add_response(url="https://api.test/r", status_code=503)
    httpx_mock.add_response(url="https://api.test/r", status_code=200, text="ok")
    fetch = make_http_fetch(
        timeout_s=2, max_retries=2, backoff_base_s=0.0, backoff_cap_s=0.0
    )
    r = fetch("https://api.test/r")
    assert r.status == 200
    assert r.text == "ok"


def test_http_adapter_exhausts_retries(httpx_mock):
    for _ in range(3):
        httpx_mock.add_response(url="https://api.test/down", status_code=503)
    fetch = make_http_fetch(
        timeout_s=2, max_retries=2, backoff_base_s=0.0, backoff_cap_s=0.0
    )
    r = fetch("https://api.test/down")
    assert r.status == 503
    assert r.ok() is False


def test_http_adapter_network_error(httpx_mock):
    import httpx

    from app.tools.adapters.http import HttpError

    httpx_mock.add_exception(httpx.ConnectError("boom"))
    fetch = make_http_fetch(
        timeout_s=2, max_retries=0, backoff_base_s=0.0, backoff_cap_s=0.0
    )
    with pytest.raises(HttpError, match="network error"):
        fetch("https://api.test/x")
