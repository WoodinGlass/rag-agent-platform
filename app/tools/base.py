"""Tool protocol.

A Tool is a name + description + JSON-schema + a callable.

Design notes:
- `call` is intentionally sync; adapters wrap async/IO if needed.
- Tool implementations should be pure where possible (calculator),
  IO-bound otherwise (web_fetch, search_docs) -> keep IO in `adapters/`.
- `input_schema` is a plain dict (JSON Schema) so it can be handed to
  any LLM provider without translation.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    func: Callable[..., Any]
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.name or not self.name.replace("_", "").isalnum():
            raise ValueError(
                f"tool name must be alnum+underscore, got {self.name!r}"
            )
        if not self.description:
            raise ValueError(f"tool {self.name!r} needs a description")
        if not isinstance(self.input_schema, dict):
            raise TypeError(
                f"tool {self.name!r} schema must be a dict, "
                f"got {type(self.input_schema).__name__}"
            )

    def call(self, **kwargs: Any) -> Any:
        return self.func(**kwargs)

    def spec(self) -> dict[str, Any]:
        """OpenAI-style function spec (provider-agnostic enough)."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.input_schema,
        }
