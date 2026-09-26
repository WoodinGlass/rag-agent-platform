"""Agent loop: plan -> tool_call -> observe -> finish.

Deterministic state machine (no LangGraph yet). The public contract is
`Agent.run(question) -> AgentOutput`. Swapping to LangGraph later must
not change this signature.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass

from pydantic import ValidationError

from app.agents.llm import LLM, AgentStep
from app.agents.prompts import initial_messages
from app.agents.registry import ToolRegistry
from app.agents.schema import AgentOutput, Citation, ToolCall
from app.core.logging import get_logger

log = get_logger(__name__)

DEFAULT_MAX_STEPS = 6


class AgentError(RuntimeError):
    """Raised when the agent cannot finish within `max_steps`."""


@dataclass
class Agent:
    llm: LLM
    registry: ToolRegistry
    max_steps: int = DEFAULT_MAX_STEPS

    def run(self, question: str) -> AgentOutput:
        question = question.strip()
        if not question:
            raise ValueError("question must be non-empty")

        messages = initial_messages(question, self.registry.all())
        tool_calls: list[ToolCall] = []

        for step_i in range(self.max_steps):
            raw = self.llm.complete(messages)
            try:
                step = AgentStep.model_validate_json(raw)
            except ValidationError as e:
                messages.append({"role": "assistant", "content": raw})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Your previous reply was invalid: {e}. "
                            f"Reply again with a single JSON object."
                        ),
                    }
                )
                log.warning("agent.invalid_step", extra={"step": step_i})
                continue

            messages.append({"role": "assistant", "content": raw})

            if step.action == "finish":
                return self._build_output(question, step, tool_calls)

            assert step.tool is not None
            tc, result = self._call_tool(step.tool, step.args)
            tool_calls.append(tc)

            observation = {
                "tool": tc.tool,
                "ok": tc.ok,
                "error": tc.error,
                "result": result,
            }
            messages.append(
                {
                    "role": "user",
                    "content": "Tool result: "
                    + json.dumps(observation, default=str),
                }
            )

        raise AgentError(f"max steps ({self.max_steps}) exceeded")

    def _call_tool(self, name: str, args: dict) -> tuple[ToolCall, object | None]:
        """Invoke a tool. Never raises; failures become ToolCall(ok=False).

        Tools are user-supplied callables; any exception they raise is
        captured here so a single bad tool cannot crash the agent loop.
        """
        t0 = time.monotonic()
        try:
            tool = self.registry.get(name)
        except KeyError as e:
            return ToolCall(tool=name, args=args, ok=False, error=str(e)), None

        try:
            result = tool.call(**args)
            ms = int((time.monotonic() - t0) * 1000)
            return ToolCall(tool=name, args=args, ok=True, latency_ms=ms), result
        except Exception as e:  # noqa: BLE001 -- tool boundary: capture anything
            ms = int((time.monotonic() - t0) * 1000)
            return (
                ToolCall(tool=name, args=args, ok=False, error=str(e), latency_ms=ms),
                None,
            )

    def _build_output(
        self,
        question: str,
        step: AgentStep,
        tool_calls: list[ToolCall],
    ) -> AgentOutput:
        cites = [Citation(**c) for c in step.citations]
        return AgentOutput(
            question=question,
            final_answer=step.final_answer or "",
            citations=cites,
            tool_calls=tool_calls,
            confidence=step.confidence,
            refused=step.refused,
            reason=step.reason,
        )
