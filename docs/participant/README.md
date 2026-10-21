# Participant Guide

## Setup

1. Install Docker Engine/Desktop + Docker Compose, or `uv` if you prefer running without Docker.
2. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. Never share this key or commit it to version control.
3. Build and run:
   - Docker: `docker compose run --rm lab-runner --help`
   - uv: `uv sync && uv run lab-cli --help`

## Safety: when does a lab actually call the Groq API?

Every lab (except Lab07, which is fully simulated and never calls the real
API) requires **both**:

1. `execution.live_api_enabled = true` in your config (`config/default.toml`
   defaults this to `false`), **and**
2. the `--live` flag on the command line.

Without both, a lab prints an explanation and exits without sending any
request or incurring any cost. This is deliberate -- running a lab with
`--live` sends real requests to the Groq API using your own key and may
incur real, if typically very small, cost on your account.

General pattern: `uv run lab-cli <lab_id> <experiment> --live [options]`.
Results are written as both JSON and CSV under `results/`.

## Lab01 -- Token Anatomy & Cost Calculation

**Objective:** see the input/output token breakdown Groq reports for a
request, and how estimated cost is computed from it.

```
uv run lab-cli lab01 all --live
```

Compare `short_prompt` vs `long_prompt`: prompt_tokens should scale with
prompt length; completion_tokens depends on what the model chooses to say.
Check `results/lab01_token_anatomy.csv` for the full breakdown and
`estimated_cost_usd` per call.

## Lab02 -- Context Growth & Context Bloat

**Objective:** observe how growing or padding the context before a
question changes prompt_tokens, cost, and latency -- while the question
and its correct answer stay the same.

```
uv run lab-cli lab02 all --live --turns 5
```

Compare `minimal_context`, `expanded_context`, and `irrelevant_context`:
only `prompt_tokens` and timing should move; the model should still answer
correctly in all three, since the relevant fact is present in every
condition. Then look at `multi_turn_growth`: prompt_tokens should increase
turn over turn as conversation history accumulates.

## Lab03 -- Prompt Caching

**Objective:** observe Groq's automatic prompt caching (currently
`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`
only) in action.

```
uv run lab-cli lab03 all --live
```

Compare `repeated_prefix` (identical long system prompt every call) vs
`cache_unfriendly` (a unique nonce prepended every call, so no exact-prefix
match is possible). Check `cached_tokens` and the printed observed cache
hit rate. **A cache miss on `repeated_prefix` is not necessarily a bug** --
caching is provider-managed, has a 2-hour TTL, and is never guaranteed.

## Lab04 -- Output Control & Invocation Optimization

**Objective:** see how `max_tokens` affects completion length/cost, and how
batching several questions into one call compares to one call per question.

```
uv run lab-cli lab04 all --live
```

For `output_limit_sweep`, completion_tokens should track the `max_tokens`
cap (until the model would have stopped earlier anyway). For
`invocation_comparison`, compare the printed `single_invocation` vs
`multiple_invocations` summaries: multiple calls repeat per-call overhead
(and, in real deployments, added network round-trip latency) that one
combined call avoids.

## Lab05 -- Model Routing

**Objective:** compare a fixed-model baseline against a simple rule-based
router that sends short/simple prompts to a small model and
longer/keyword-flagged prompts to a large model, over a small golden
dataset with deterministic pass/fail checks.

```
uv run lab-cli lab05 all --live
```

Compare the two `summary` lines printed (accuracy / cost / latency).
The routing heuristic is intentionally simple (prompt length + a keyword
list) and is not guaranteed to generalize beyond this dataset -- that
limitation is itself worth discussing.

## Lab06 -- Streaming & Perceived Latency

**Objective:** distinguish time-to-first-token (TTFT) from
time-to-first-*visible*-content, and compare streaming vs non-streaming
total latency and throughput.

```
uv run lab-cli lab06 all --live
```

`ttft_ms` marks the first network chunk; `first_visible_content_ms` marks
the first chunk that actually carries visible text -- these can differ if
the first chunk only carries role metadata. Token usage is intentionally
not reported for the streaming condition (Groq's streaming usage support
was not confirmed during this project's research); `output_tokens_per_second`
instead comes from the separate non-streaming call.

## Lab07 -- Rate Limiting & Retry Strategies (fully simulated)

**Objective:** compare retry strategies (no backoff, exponential backoff,
exponential backoff with jitter, retry-after-aware) under a controlled,
simulated sequence of HTTP 429 responses.

```
uv run lab-cli lab07 all
```

This lab never calls the real Groq API -- it is explicitly simulation-based,
per the project's policy against deliberately exceeding real rate limits.
Compare `total_wait_seconds` and `retry_count` across strategies.

## Lab08 -- Async Processing & Concurrency

**Objective:** compare sequential (one request at a time) vs concurrent
(`asyncio.gather` with a semaphore limit) execution of the same batch of
prompts.

```
uv run lab-cli lab08 all --live --max-concurrent 4
```

Compare total wall-clock time, `requests_per_second`, and p50/p95 latency
between `sequential` and `concurrent`. Concurrency should reduce total time
but not necessarily per-call latency.

## Lab09 -- Batch Processing

**Objective:** contrast three distinct concepts: synchronous requests,
Python-level async/concurrent requests, and the provider-managed Groq
Batch API.

```
uv run lab-cli lab09 sequential --live
uv run lab-cli lab09 async --live
uv run lab-cli lab09 prepare_batch_input          # no --live needed, pure local file write
uv run lab-cli lab09 submit_batch --live           # submits only; does NOT wait
uv run lab-cli lab09 check_batch_status --live --batch-id <id>
uv run lab-cli lab09 fetch_batch_results --live --output-file-id <id>
```

**A real batch job's completion window is 24 hours to 7 days** -- it will
not finish during a class session. See
`docs/instructor/lab09_sample_batch_result.json` for a pre-captured sample
to discuss the Batch API's output shape and cost trade-off without waiting.

## Troubleshooting

- `GROQ_API_KEY is not set`: export it (`.env` via Docker Compose, or your
  shell environment when using `uv` directly).
- `Live execution requested but execution.live_api_enabled is false`: edit
  your config's `[execution]` section, or pass `--config <path>` to a copy
  with it set to `true`.
- Results look identical across repetitions: real models and networks are
  non-deterministic; run with more `--repetitions` before drawing conclusions
  from a single call.
