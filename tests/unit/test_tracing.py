"""Tracing: noop path (default) + enabled path (skipped if SDK absent).

Offline; no network. Enabled path only runs if opentelemetry-sdk is
installed (it is in CI via the [otel] extra).
"""
import pytest

from app.core.tracing import (
    NoopSpan,
    _coerce,
    configure_tracing,
    is_enabled,
    span,
)

# ---------- coercion ----------

def test_coerce_primitives():
    assert _coerce("x") == "x"
    assert _coerce(1) == 1
    assert _coerce(1.5) == 1.5
    assert _coerce(True) is True
    assert _coerce(None) is None


def test_coerce_list_to_str_list():
    assert _coerce([1, 2, 3]) == ["1", "2", "3"]


def test_coerce_dict_to_str():
    assert _coerce({"a": 1}) == "{'a': 1}"


# ---------- noop path ----------

def test_disabled_by_default(monkeypatch):
    monkeypatch.setattr("app.core.tracing._enabled", False, raising=False)
    configure_tracing(enabled=False)
    assert is_enabled() is False


def test_noop_span_usable_when_disabled():
    configure_tracing(enabled=False)
    with span("test.op", k=1) as s:
        assert isinstance(s, NoopSpan)
        s.set_attribute("x", 1)          # no error
        s.add_event("evt", {"a": 1})     # no error
        s.record_exception(ValueError()) # no error


def test_span_does_not_swallow_exceptions_when_disabled():
    configure_tracing(enabled=False)
    with pytest.raises(RuntimeError, match="boom"), span("test.op"):
        raise RuntimeError("boom")


# ---------- enabled path (needs OTel SDK) ----------

otel = pytest.importorskip(
    "opentelemetry.sdk.trace",
    reason="opentelemetry-sdk not installed; run: pip install -e '.[otel]'",
)


def test_enabled_with_sdk_sets_flag():
    configure_tracing(
        enabled=True,
        service_name="test-svc",
        exporter="otlp",
        otlp_endpoint="http://127.0.0.1:9/v1/traces",
    )
    assert is_enabled() is True
    # restore for other tests
    configure_tracing(enabled=False)


def test_enabled_span_yields_usable_object():
    configure_tracing(
        enabled=True,
        service_name="test-svc",
        exporter="otlp",
        otlp_endpoint="http://127.0.0.1:9/v1/traces",
    )
    try:
        with span("test.enabled", k=1, tenant="acme") as s:
            s.set_attribute("answer", 42)
            s.add_event("checkpoint")
        assert not isinstance(s, NoopSpan)
    finally:
        configure_tracing(enabled=False)


def test_enabled_span_records_exception_and_reraises():
    configure_tracing(
        enabled=True,
        service_name="test-svc",
        exporter="otlp",
        otlp_endpoint="http://127.0.0.1:9/v1/traces",
    )
    try:
        with pytest.raises(ValueError, match="kaboom"), span("test.error"):
            raise ValueError("kaboom")
    finally:
        configure_tracing(enabled=False)

# ---------- exporter selection (M6.3) ----------

def test_build_exporter_console():
    from app.core.tracing import _build_exporter

    exp = _build_exporter("console", "unused")
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    assert isinstance(exp, ConsoleSpanExporter)


def test_build_exporter_otlp():
    from app.core.tracing import _build_exporter

    exp = _build_exporter("otlp", "http://example.test:4318/v1/traces")
    # OTLPSpanExporter has an ._endpoint attribute
    assert "example.test" in str(getattr(exp, "_endpoint", ""))


def test_build_exporter_unknown_raises():
    from app.core.tracing import _build_exporter

    with pytest.raises(ValueError, match="unknown otel exporter"):
        _build_exporter("bogus", "x")


def test_configure_tracing_otlp_end_to_end():
    """OTLP path activates without connecting anywhere."""
    configure_tracing(
        enabled=True,
        service_name="test-otlp",
        exporter="otlp",
        otlp_endpoint="http://127.0.0.1:9/unreachable",
    )
    try:
        assert is_enabled() is True
    finally:
        configure_tracing(enabled=False)

