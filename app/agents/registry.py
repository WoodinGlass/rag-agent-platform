"""Tool registry.

Pattern: register by name, resolve by name. No `if name == "x"` chains.
"""
from __future__ import annotations

from app.tools.base import Tool


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError as e:
            raise KeyError(
                f"unknown tool {name!r}; have: {sorted(self._tools)}"
            ) from e

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def all(self) -> list[Tool]:
        return [self._tools[n] for n in self.names()]

    def specs(self) -> list[dict]:
        return [t.spec() for t in self.all()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


_default = ToolRegistry()


def default_registry() -> ToolRegistry:
    return _default


def register(tool: Tool) -> None:
    _default.register(tool)


def get(name: str) -> Tool:
    return _default.get(name)
