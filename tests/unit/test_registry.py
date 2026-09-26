import pytest

from app.agents.registry import ToolRegistry
from app.tools.base import Tool


def _tool(name: str) -> Tool:
    return Tool(
        name=name,
        description=f"{name} tool",
        input_schema={"type": "object", "properties": {}},
        func=lambda: name,
    )


def test_register_and_get():
    r = ToolRegistry()
    r.register(_tool("a"))
    assert r.get("a").call() == "a"


def test_duplicate_register_rejected():
    r = ToolRegistry()
    r.register(_tool("a"))
    with pytest.raises(ValueError, match="already registered"):
        r.register(_tool("a"))


def test_get_unknown_raises_keyerror():
    r = ToolRegistry()
    with pytest.raises(KeyError, match="unknown tool"):
        r.get("nope")


def test_has_and_contains():
    r = ToolRegistry()
    r.register(_tool("a"))
    assert r.has("a") is True
    assert "a" in r
    assert "b" not in r


def test_names_sorted_and_len():
    r = ToolRegistry()
    for n in ("c", "a", "b"):
        r.register(_tool(n))
    assert r.names() == ["a", "b", "c"]
    assert len(r) == 3


def test_all_and_specs_aligned():
    r = ToolRegistry()
    for n in ("b", "a"):
        r.register(_tool(n))
    tools = r.all()
    specs = r.specs()
    assert [t.name for t in tools] == ["a", "b"]
    assert [s["name"] for s in specs] == ["a", "b"]


def test_unregister():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.unregister("a")
    assert "a" not in r
    r.unregister("a")


def test_unknown_tool_error_lists_available():
    r = ToolRegistry()
    r.register(_tool("a"))
    r.register(_tool("b"))
    with pytest.raises(KeyError, match=r"\['a', 'b'\]"):
        r.get("z")
