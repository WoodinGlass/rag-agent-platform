import pytest

from app.tools.base import Tool


def _echo(x: int) -> int:
    return x


def _make(**over):
    base = {
        "name": "echo",
        "description": "echo an int",
        "input_schema": {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        },
        "func": _echo,
    }
    base.update(over)
    return Tool(**base)


def test_call_forwards_kwargs():
    assert _make().call(x=7) == 7


def test_spec_shape():
    spec = _make().spec()
    assert spec["name"] == "echo"
    assert spec["description"] == "echo an int"
    assert spec["parameters"]["type"] == "object"


def test_spec_is_openai_compatible():
    spec = _make().spec()
    assert set(spec) == {"name", "description", "parameters"}


def test_bad_name_rejected():
    with pytest.raises(ValueError):
        _make(name="bad name!")


def test_empty_name_rejected():
    with pytest.raises(ValueError):
        _make(name="")


def test_empty_description_rejected():
    with pytest.raises(ValueError):
        _make(description="")


def test_non_dict_schema_rejected():
    with pytest.raises(TypeError):
        _make(input_schema=["not", "a", "dict"])


def test_tags_default_empty():
    assert _make().tags == ()


def test_tags_passed():
    assert _make(tags=("pure",)).tags == ("pure",)
