"""Tracing: OpenTelemetry when enabled, no-op otherwise.

Design:
- Always yield a span-like object with the same tiny interface, so call
  sites look identical whether tracing is on or off.
- OTel SDK is imported lazily — enabling requires the `[otel]` extra.
  If enable is requested but the SDK is missing, we log a warning and
  stay disabled. Never break the app because of tracing.
- Two exporters: "console" (dev) and "otlp" (send to a collector).
  Selection is via config; missing exporter package -> warning + disabled.
- Configure once at startup (`app/main.py` lifespan).
- Span attribute values are coerced to OTel's accepted primitives.
"""
from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

log = logging.getLogger(__name__)

_service_name = "rag-agent-platform"
_enabled = False
_configured = False
_exporter_kind = "console"
_otlp_endpoint = "http://localhost:4318/v1/traces"


class Span:
    """Minimal span interface. Both impls expose the same methods."""

    def set_attribute(self, key: str, value: Any) -> None:  # pragma: no cover
        raise NotImplementedError

    def add_event(self, name: str, attrs: dict[str, Any] | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def record_exception(self, exc: BaseException) -> None:  # pragma: no cover
        raise NotImplementedError


class NoopSpan(Span):
    def set_attribute(self, key: str, value: Any) -> None:
        return None

    def add_event(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        return None

    def record_exception(self, exc: BaseException) -> None:
        return None


class OTelSpan(Span):
    def __init__(self, otel_span: Any) -> None:
        self._s = otel_span

    def set_attribute(self, key: str, value: Any) -> None:
        try:
            self._s.set_attribute(key, value)
        except Exception:
            log.debug("tracing: failed to set attribute %s", key, exc_info=True)

    def add_event(self, name: str, attrs: dict[str, Any] | None = None) -> None:
        try:
            self._s.add_event(name, attrs or {})
        except Exception:
            log.debug("tracing: failed to add event %s", name, exc_info=True)

    def record_exception(self, exc: BaseException) -> None:
        try:
            self._s.record_exception(exc)
        except Exception:
            log.debug("tracing: failed to record exception", exc_info=True)


def _coerce(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [str(v) for v in value]
    return str(value)


def is_enabled() -> bool:
    return _enabled


def _build_exporter(kind: str, otlp_endpoint: str) -> Any:
    """Return an exporter instance. Raises ImportError or ValueError."""
    from opentelemetry.sdk.trace.export import ConsoleSpanExporter

    if kind == "console":
        return ConsoleSpanExporter()

    if kind == "otlp":
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise ImportError(
                "otlp exporter not installed; run: "
                "pip install 'rag-agent-platform[otel]'"
            ) from e
        return OTLPSpanExporter(endpoint=otlp_endpoint)

    raise ValueError(f"unknown otel exporter: {kind!r}")


def configure_tracing(
    *,
    enabled: bool = False,
    service_name: str | None = None,
    exporter: str = "console",
    otlp_endpoint: str = "http://localhost:4318/v1/traces",
) -> None:
    """Enable/disable tracing. Safe to call multiple times. Never raises."""
    global _service_name, _enabled, _configured, _exporter_kind, _otlp_endpoint

    if service_name:
        _service_name = service_name
    _exporter_kind = exporter
    _otlp_endpoint = otlp_endpoint

    if not enabled:
        _enabled = False
        return

    if _configured:
        _enabled = True
        return

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    except ImportError:
        log.warning(
            "OTEL_ENABLED=true but opentelemetry-sdk is not installed; "
            "install with: pip install 'rag-agent-platform[otel]'"
        )
        _enabled = False
        return

    try:
        exporter_instance = _build_exporter(exporter, otlp_endpoint)
    except (ImportError, ValueError) as e:
        log.warning("tracing disabled: %s", e)
        _enabled = False
        return

    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter_instance))
    trace.set_tracer_provider(provider)

    _configured = True
    _enabled = True
    log.info(
        "tracing enabled: service=%s exporter=%s endpoint=%s",
        _service_name,
        exporter,
        otlp_endpoint if exporter == "otlp" else "-",
    )


@contextmanager
def span(name: str, **attrs: Any) -> Iterator[Span]:
    """Start a span. No-op when tracing is disabled."""
    if not _enabled:
        yield NoopSpan()
        return

    from opentelemetry import trace

    tracer = trace.get_tracer(_service_name)
    with tracer.start_as_current_span(name) as otel_span:
        for k, v in attrs.items():
            cv = _coerce(v)
            if cv is not None:
                otel_span.set_attribute(k, cv)
        wrapper = OTelSpan(otel_span)
        try:
            yield wrapper
        except Exception as e:
            wrapper.record_exception(e)
            try:
                from opentelemetry.trace import Status, StatusCode
                otel_span.set_status(Status(StatusCode.ERROR, str(e)))
            except Exception:
                log.debug("tracing: failed to set error status", exc_info=True)
            raise
