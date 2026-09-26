"""LangGraph-backed agent loop.

Same contract, same behavior as `Agent` (the state machine in
`app/agents/agent.py`). Chosen by `AGENT_BACKEND=langgraph`.

Graph shape:

    plan -> (tool_call -> observe)* -> finish

Where:
- `plan`       calls the LLM, parses `AgentStep`, routes.
- `execute`    runs the chosen tool, appends observation to state.
- `finish`     builds `AgentOutput` from the last step.

The heavy lifting (parse, call, guard) is shared with the state machine:
we import `AgentStep`, `ToolCall`, `AgentOutput` — no duplication of
schema or business rules.

LangGraph import is lazy: the module only requires the package when the
backend is actually selected at runtime.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from app.agents.llm import LLM, AgentStep
from app.agents.prompts import initial_messages
from app.agents.registry import ToolRegistry
from app.agents.schema import AgentOutput, Citation, ToolCall
from app.core.logging import get_logger
from app.core.tracing import span

log = get_logger(__name__)

DEFAULT_MAX_STEPS = 6


class LangGraphError(RuntimeError):
    """Raised when the graph cannot finish within `max_steps`."""


@dataclass
class _GraphState:
    """Mutable state passed between nodes. LangGraph merges updates into it."""

    question: str
    messages: list[dict] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    next_action: str = ""       # "" | "tool" | "finish"
    last_step: dict | None = None
    steps: int = 0


class LangGraphAgent:
    """Agent backend backed by a `langgraph.graph.StateGraph`.

    Public contract matches `app.agents.agent.Agent` (see `AgentBackend`).
    """

    def __init__(
        self,
        llm: LLM,
        registry: ToolRegistry,
        max_steps: int = DEFAULT_MAX_STEPS,
    ) -> None:
        try:
            from langgraph.graph import END, StateGraph
        except ImportError as e:  # pragma: no cover - optional dep
            raise ImportError(
                "langgraph not installed; run: pip install 'rag-agent-platform[langgraph]'"
            ) from e

        self.llm = llm
        self.registry = registry
        self.max_steps = max_steps

        builder = StateGraph(_GraphState)
        builder.add_node("plan", self._node_plan)
        builder.add_node("execute", self._node_execute)
        builder.add_node("finish", self._node_finish)

        builder.set_entry_point("plan")
        builder.add_conditional_edges(
            "plan",
            self._route_from_plan,
            {"execute": "execute", "finish": "finish", "plan": "plan"},
        )
        builder.add_edge("execute", "plan")
        builder.add_edge("finish", END)

        self._graph = builder.compile()

    # ---------- public API ----------

    def run(self, question: str) -> AgentOutput:
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        with span("agent.run", max_steps=self.max_steps, backend="langgraph") as s:
            state = _GraphState(
                question=question,
                messages=initial_messages(question, self.registry.all()),
            )
            result = self._graph.invoke(
                state,
                config={"recursion_limit": max(10, self.max_steps * 3)},
            )
            out = self._build_output(
                question,
                result.get("last_step") or {},
                result.get("tool_calls") or [],
            )
            s.set_attribute("n_tool_calls", len(out.tool_calls))
            s.set_attribute("refused", out.refused)
            s.set_attribute("confidence", out.confidence)
            return out

    # ---------- routing ----------

    def _route_from_plan(self, state: _GraphState) -> str:
        if state.next_action == "finish":
            return "finish"
        if state.next_action == "tool":
            return "execute"
        # Invalid step / parse failure -> retry the plan node.
        return "plan"

    # ---------- nodes ----------

    def _node_plan(self, state: _GraphState) -> dict[str, Any]:
        """Call the LLM, parse one step, decide the next action."""
        if state.steps >= self.max_steps:
            # Loop cap reached -> synthesize a refusal
            return {
                "next_action": "finish",
                "last_step": {
                    "final_answer": "stopped: max steps reached",
                    "citations": [],
                    "confidence": "low",
                    "refused": True,
                    "reason": f"max steps ({self.max_steps}) exceeded",
                },
                "steps": state.steps,
            }

        raw = self.llm.complete(state.messages)
        try:
            step = AgentStep.model_validate_json(raw)
        except ValidationError as e:
            # Feed the parse error back so the LLM can self-correct.
            return {
                "messages": state.messages
                + [
                    {"role": "assistant", "content": raw},
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {e}. "
                            "Reply again with a single JSON object."
                        ),
                    },
                ],
                "next_action": "retry",  # loop back to plan
                "steps": state.steps + 1,
            }

        messages_next = state.messages + [{"role": "assistant", "content": raw}]

        if step.action == "finish":
            return {
                "messages": messages_next,
                "next_action": "finish",
                "last_step": step.model_dump(),
                "steps": state.steps + 1,
            }

        # action == "tool": route to execute
        return {
            "messages": messages_next,
            "next_action": "tool",
            "last_step": step.model_dump(),
            "steps": state.steps + 1,
        }

    def _node_execute(self, state: _GraphState) -> dict[str, Any]:
        """Run the tool named in the last step; append observation."""
        step = state.last_step or {}
        name = step.get("tool") or ""
        args = step.get("args") or {}

        tc, result = self._call_tool(name, args)
        observation = {
            "tool": tc.tool,
            "ok": tc.ok,
            "error": tc.error,
            "result": result,
        }
        messages_next = state.messages + [
            {
                "role": "user",
                "content": "Tool result: "
                + json.dumps(observation, default=str),
            }
        ]
        return {
            "messages": messages_next,
            "tool_calls": state.tool_calls + [tc],
        }

    def _node_finish(self, state: _GraphState) -> dict[str, Any]:
        # No-op node — the actual output is built in `run()` from state.
        return {}

    # ---------- tool invocation (same guard as Agent) ----------

    def _call_tool(self, name: str, args: dict) -> tuple[ToolCall, object | None]:
        with span(f"tool.{name}", tool=name) as s:
            t0 = time.monotonic()
            try:
                tool = self.registry.get(name)
            except KeyError as e:
                s.set_attribute("ok", False)
                s.set_attribute("error", "unknown tool")
                return ToolCall(tool=name, args=args, ok=False, error=str(e)), None

            try:
                result = tool.call(**args)
                ms = int((time.monotonic() - t0) * 1000)
                s.set_attribute("ok", True)
                s.set_attribute("latency_ms", ms)
                return ToolCall(tool=name, args=args, ok=True, latency_ms=ms), result
            except Exception as e:  # noqa: BLE001 -- tool boundary
                ms = int((time.monotonic() - t0) * 1000)
                s.set_attribute("ok", False)
                s.set_attribute("latency_ms", ms)
                s.set_attribute("error", str(e)[:200])
                s.record_exception(e)
                return (
                    ToolCall(tool=name, args=args, ok=False, error=str(e), latency_ms=ms),
                    None,
                )

    # ---------- output ----------

    def _build_output(
        self,
        question: str,
        last_step: dict,
        tool_calls: list[ToolCall],
    ) -> AgentOutput:
        cites = [Citation(**c) for c in (last_step.get("citations") or [])]
        return AgentOutput(
            question=question,
            final_answer=last_step.get("final_answer") or "",
            citations=cites,
            tool_calls=tool_calls,
            confidence=last_step.get("confidence", "medium"),
            refused=bool(last_step.get("refused", False)),
            reason=last_step.get("reason", ""),
        )
