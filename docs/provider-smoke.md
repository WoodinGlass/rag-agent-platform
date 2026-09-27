# Provider smoke test

**What this proves:** the real LLM provider path (not just `FakeLLM`)
actually works against a live API — the wrapper, the prompt pattern, and
the agent loop all survive contact with a real model.

**How to run:**

```bash
# Groq (free tier, OpenAI-compatible)
GROQ_API_KEY=gsk_... pytest -m provider -v

# OpenAI (also enables the embeddings tests)
OPENAI_API_KEY=sk-... pytest -m provider -v
```

Or via GitHub Actions: **Provider smoke** workflow (manual + weekly cron).
It self-skips when neither key is set as a repository secret, so it never
blocks the offline CI.

---

## Last verified run

| Field | Value |
|---|---|
| Date | 2026-09-27 |
| Provider | Groq (`https://api.groq.com/openai/v1`) |
| Model | `qwen/qwen3.8-27b` |
| Result | **4 passed, 2 skipped** |
| Skipped | the two `OpenAIEmbedder` tests (Groq has no embeddings endpoint) |
| Cost | $0.00 (Groq free tier) |

```
tests/integration/test_provider_smoke.py
[provider-smoke] using: groq model=qwen/qwen3.8-27b
....ss                                                          [100%]
```

Tests that passed:

1. `test_provider_reports_which_key_is_active` — informational
2. `test_llm_emits_parseable_json_finish` — model emits valid `AgentStep` JSON
3. `test_llm_emits_tool_call_when_asked` — model emits a `{"action":"tool"}` step
4. `test_agent_run_with_real_llm_returns_valid_output` — full agent loop
   returns a valid `AgentOutput`

Tests that skipped:

5. `test_openai_embedder_returns_vector_of_expected_dim` — needs `OPENAI_API_KEY`
6. `test_openai_embedder_batch_returns_one_per_input` — needs `OPENAI_API_KEY`

---

## Model selection: why `qwen/qwen3.8-27b`

Choosing a provider is the easy part. Choosing a model is where the
interesting failures live. We tested three Groq models before settling:

| Model | Result | Why |
|---|---|---|
| `openai/gpt-oss-20b` | FAIL | Native tool-calling. When asked for a JSON object, it emits the OpenAI function-call shape (`{"name": ..., "arguments": ...}`) instead of our `{"action": "tool", "tool": ...}`. Groq's API then rejects it: `Tool choice is none, but model called a tool`. |
| `allam-2-7b` | FAIL | Too small for the multi-turn agent loop. Handles a single JSON reply, but as soon as `messages` grows (tool result + error feedback), it fails parsing repeatedly and trips `max_steps`. |
| `qwen/qwen3.8-27b` | PASS | Emits clean JSON, follows the tool-call schema, and survives the multi-turn loop. |

Additional constraint: Groq's free tier caps **output tokens per minute
(OTPM)** at 1000. `qwen3.8-27b` initially requested 1295 output tokens
and hit a `RateLimitError`. We cap `max_tokens=256` in `OpenAILLM` (and
the Groq branch of `main.py`), which keeps every request well under the
cap and matches our outputs (short structured JSON, not long prose).

---

## What this does **not** prove

- **Production robustness.** One pass over six tests is a smoke test,
  not a load test. Latency, cost, and failure rates under load are
  separate questions.
- **Model quality.** We assert that the model emits our schema, not that
  the answers are good. Quality is measured by `evals/` (Ragas), not
  here.
- **All providers equally.** Groq is validated today. OpenAI and
  Anthropic share the same `LLM` protocol; their wrappers are
  unit-tested against mocks (`tests/unit/test_llm_openai.py`), and the
  smoke test runs them too when their keys are present.

---

## Related

- Wrapper: `app/agents/llm.py`
- Prompt pattern that assumes a JSON emitter, not a native tool-caller:
  `app/agents/prompts.py`
- Offline unit tests for the wrappers: `tests/unit/test_llm_openai.py`
- Workflow: `.github/workflows/provider-smoke.yml`
