import pytest

from app.tools.calculator import evaluate, make_calculator_tool


def _calc(expr):
    return make_calculator_tool().call(expression=expr)


# ---------- happy path ----------

def test_basic_addition():
    assert _calc("1+2")["result"] == 3


def test_operator_precedence():
    assert _calc("2+3*4")["result"] == 14


def test_parentheses():
    assert _calc("(2+3)*4")["result"] == 20


def test_unary_minus():
    assert _calc("-5+3")["result"] == -2


def test_power():
    assert _calc("2**10")["result"] == 1024


def test_true_division():
    assert _calc("10/4")["result"] == 2.5


def test_floor_division():
    assert _calc("10//3")["result"] == 3


def test_modulo():
    assert _calc("10%3")["result"] == 1


def test_float_literal():
    assert abs(_calc("0.1+0.2")["result"] - 0.3) < 1e-9


def test_result_repr_present():
    out = _calc("1+1")
    assert out["expression"] == "1+1"
    assert out["result_repr"] == "2"


# ---------- injection attempts ----------

def test_reject_name():
    with pytest.raises(ValueError, match="not allowed"):
        _calc("x + 1")


def test_reject_call():
    with pytest.raises(ValueError, match="not allowed"):
        _calc("__import__('os')")


def test_reject_attribute():
    with pytest.raises(ValueError, match="not allowed"):
        _calc("(1).__class__")


def test_reject_subscript():
    with pytest.raises(ValueError, match="not allowed"):
        _calc("[1,2][0]")


def test_reject_lambda():
    with pytest.raises(ValueError, match="not allowed"):
        _calc("lambda x: x")


def test_reject_string_literal():
    with pytest.raises(TypeError, match="numeric"):
        _calc("'hello'")


def test_reject_bool_literal():
    with pytest.raises(TypeError, match="numeric"):
        _calc("True")


# ---------- input validation ----------

def test_reject_empty():
    with pytest.raises(ValueError, match="non-empty"):
        _calc("")


def test_reject_whitespace():
    with pytest.raises(ValueError, match="non-empty"):
        _calc("   ")


def test_reject_non_string():
    with pytest.raises(TypeError):
        evaluate(123)  # type: ignore[arg-type]


def test_reject_syntax_error():
    with pytest.raises(ValueError, match="invalid expression"):
        _calc("1 +")


# ---------- DoS caps ----------

def test_reject_exponent_dos():
    with pytest.raises(ValueError, match="exponent too large"):
        _calc("2**9999")


def test_reject_huge_result():
    with pytest.raises(ValueError, match="magnitude too large"):
        _calc("10**500")


# ---------- runtime errors from Python itself ----------

def test_division_by_zero():
    with pytest.raises(ZeroDivisionError):
        _calc("1/0")


# ---------- tool metadata ----------

def test_spec_shape():
    t = make_calculator_tool()
    assert t.name == "calculator"
    assert t.spec()["parameters"]["required"] == ["expression"]
    assert "pure" in t.tags
