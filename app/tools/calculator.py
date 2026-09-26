"""Safe arithmetic evaluator tool.

Pure logic; no IO, no adapter needed.

Design:
- Uses Python's `ast` module to whitelist node types.
- NEVER uses `eval()`. Injection attempts are rejected by node type.
- DoS caps: exponent limit + result magnitude limit.

Allowed:
  - numeric literals (int, float) — bool is explicitly rejected
  - binary ops: + - * / // % **
  - unary ops: +x, -x
  - parentheses (implicit via AST)

Rejected (ValueError/TypeError, "not allowed"/"numeric"):
  - names, function calls, attribute access, subscripts
  - comprehensions, lambdas, conditionals, comparisons
  - string literals (TypeError), bool literals (TypeError)
"""
from __future__ import annotations

import ast
import operator
from collections.abc import Callable
from typing import Any

from app.tools.base import Tool

MAX_EXPONENT = 1000
MAX_MAGNITUDE = 1e100

_BIN_OPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "expression": {
            "type": "string",
            "description": "Arithmetic expression, e.g. '(2+3)*4'.",
        },
    },
    "required": ["expression"],
}


def _eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(
            node.value, (int, float)
        ):
            raise TypeError(
                f"only numeric literals allowed, got {type(node.value).__name__}"
            )
        return node.value

    if isinstance(node, ast.BinOp):
        bin_op = _BIN_OPS.get(type(node.op))
        if bin_op is None:
            raise ValueError(f"operator {type(node.op).__name__} not allowed")
        left = _eval(node.left)
        right = _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"exponent too large (max {MAX_EXPONENT})")
        result = bin_op(left, right)
        if abs(result) > MAX_MAGNITUDE:
            raise ValueError(
                f"result magnitude too large (max {MAX_MAGNITUDE:g})"
            )
        return result

    if isinstance(node, ast.UnaryOp):
        unary_op = _UNARY_OPS.get(type(node.op))
        if unary_op is None:
            raise ValueError(
                f"unary operator {type(node.op).__name__} not allowed"
            )
        return unary_op(_eval(node.operand))

    raise ValueError(f"node {type(node).__name__} not allowed")


def evaluate(expression: str) -> int | float:
    """Evaluate a safe arithmetic expression. Raises on anything unsafe."""
    if not isinstance(expression, str):
        raise TypeError(
            f"expression must be str, got {type(expression).__name__}"
        )
    if not expression.strip():
        raise ValueError("expression must be a non-empty string")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid expression: {e.msg}") from e
    return _eval(tree)


def make_calculator_tool(*, name: str = "calculator") -> Tool:
    """Build the calculator Tool (no external dependency)."""

    def _calc(expression: str) -> dict:
        result = evaluate(expression)
        return {
            "expression": expression,
            "result": result,
            "result_repr": repr(result),
        }

    return Tool(
        name=name,
        description=(
            "Evaluate a basic arithmetic expression. Supports "
            "+ - * / // % ** and parentheses. No variables or functions."
        ),
        input_schema=INPUT_SCHEMA,
        func=_calc,
        tags=("pure", "math"),
    )
