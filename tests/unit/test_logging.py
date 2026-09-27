"""Tests for app.core.logging: correlation id + formatters + config.

Offline and deterministic: no reliance on the process's logging config,
no real handlers left on the root logger after each test.
"""
from __future__ import annotations

import json
import logging
import sys

from app.core.logging import (
    JsonFormatter,
    PlainFormatter,
    configure_logging,
    get_correlation_id,
    get_logger,
    new_correlation_id,
    set_correlation_id,
)

# ---------- helpers ----------

def _record(
    msg="hello",
    *,
    level=logging.INFO,
    name="test.logger",
    extra: dict | None = None,
    exc_info=None,
    args=(),
):
    rec = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=args,
        exc_info=exc_info,
    )
    if extra:
        for k, v in extra.items():
            setattr(rec, k, v)
    return rec


def _reset_root() -> None:
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)


# ---------- correlation id ----------

def test_new_correlation_id_is_hex_and_short():
    cid = new_correlation_id()
    assert len(cid) == 16
    assert all(c in "0123456789abcdef" for c in cid)


def test_new_correlation_id_is_unique():
    assert new_correlation_id() != new_correlation_id()


def test_new_correlation_id_sets_contextvar():
    cid = new_correlation_id()
    assert get_correlation_id() == cid


def test_set_and_get_round_trip():
    set_correlation_id("abc123")
    try:
        assert get_correlation_id() == "abc123"
    finally:
        set_correlation_id("")


def test_default_is_empty():
    set_correlation_id("")
    assert get_correlation_id() == ""


# ---------- JsonFormatter ----------

def test_json_formatter_outputs_valid_json_with_required_fields():
    set_correlation_id("cid-1")
    out = JsonFormatter().format(_record("hi"))
    payload = json.loads(out)
    assert payload["msg"] == "hi"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logger"
    assert payload["cid"] == "cid-1"
    assert isinstance(payload["ts"], float)


def test_json_formatter_includes_custom_extras():
    rec = _record(extra={"doc_id": "d1", "n_chunks": 3})
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["doc_id"] == "d1"
    assert payload["n_chunks"] == 3


def test_json_formatter_skips_private_and_reserved_keys():
    rec = _record(extra={"_private": "x", "module": "override"})
    payload = json.loads(JsonFormatter().format(rec))
    # private key dropped
    assert "_private" not in payload
    # reserved key never leaks from extra (it is filtered out entirely)
    assert "module" not in payload


def test_json_formatter_serializes_unusual_objects_via_default():
    class Weird:
        def __repr__(self) -> str:
            return "<weird>"

    rec = _record(extra={"obj": Weird()})
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["obj"] == "<weird>"


def test_json_formatter_handles_exception():
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        rec = _record(exc_info=sys.exc_info())
    payload = json.loads(JsonFormatter().format(rec))
    assert "exc" in payload
    assert "RuntimeError" in payload["exc"]
    assert "boom" in payload["exc"]


def test_json_formatter_formats_msg_with_args():
    rec = _record(msg="hello %s", args=("world",))
    payload = json.loads(JsonFormatter().format(rec))
    assert payload["msg"] == "hello world"


def test_json_formatter_preserves_non_ascii():
    rec = _record(msg="héllo — emoji 🎉")
    out = JsonFormatter().format(rec)
    # ensure_ascii=False -> literal characters preserved
    assert "héllo" in out
    assert "🎉" in out


def test_json_formatter_uses_current_contextvar_at_format_time():
    set_correlation_id("first")
    fmt = JsonFormatter()
    rec = _record()
    set_correlation_id("second")
    payload = json.loads(fmt.format(rec))
    # cid reflects the state at format() time, not at record creation
    assert payload["cid"] == "second"


# ---------- PlainFormatter ----------

def test_plain_formatter_with_cid_prefix():
    set_correlation_id("cid-2")
    out = PlainFormatter().format(_record("hello"))
    assert out.startswith("[cid-2] ")
    assert "INFO test.logger: hello" in out


def test_plain_formatter_without_cid():
    set_correlation_id("")
    out = PlainFormatter().format(_record("hello"))
    assert not out.startswith("[")
    assert out == "INFO test.logger: hello"


def test_plain_formatter_error_level_included():
    set_correlation_id("")
    out = PlainFormatter().format(_record("bad", level=logging.ERROR, name="x"))
    assert out == "ERROR x: bad"


# ---------- configure_logging ----------

def test_configure_logging_json_sets_handler_and_level():
    try:
        configure_logging(level="DEBUG", json_output=True)
        root = logging.getLogger()
        assert root.level == logging.DEBUG
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, JsonFormatter)
    finally:
        _reset_root()


def test_configure_logging_plain_sets_handler_and_level():
    try:
        configure_logging(level="warning", json_output=False)
        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert len(root.handlers) == 1
        assert isinstance(root.handlers[0].formatter, PlainFormatter)
    finally:
        _reset_root()


def test_configure_logging_clears_previous_handlers():
    root = logging.getLogger()
    root.addHandler(logging.NullHandler())
    root.addHandler(logging.NullHandler())
    try:
        configure_logging(level="INFO", json_output=True)
        assert len(root.handlers) == 1
    finally:
        _reset_root()


def test_configure_logging_level_is_case_insensitive():
    try:
        configure_logging(level="debug", json_output=True)
        assert logging.getLogger().level == logging.DEBUG
    finally:
        _reset_root()


def test_configure_logging_emits_record_through_formatter(capsys):
    try:
        configure_logging(level="INFO", json_output=True)
        logging.getLogger("t").info("emitted")
        captured = capsys.readouterr()
        # one JSON line on stdout
        payload = json.loads(captured.out.strip().splitlines()[-1])
        assert payload["msg"] == "emitted"
    finally:
        _reset_root()


# ---------- get_logger ----------

def test_get_logger_returns_named_logger():
    lg = get_logger("app.foo")
    assert isinstance(lg, logging.Logger)
    assert lg.name == "app.foo"


def test_get_logger_same_name_same_instance():
    assert get_logger("x") is get_logger("x")
