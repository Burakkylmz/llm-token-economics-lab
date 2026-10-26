# Participant Guide

## Setup

1. Install Docker Engine/Desktop + Docker Compose (recommended), or `uv` if
   you prefer running without Docker.
2. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`. Never
   share this key or commit it to version control.
3. **Create your own live config.** `config/default.toml` ships with
   `execution.live_api_enabled = false` on purpose (see **Safety**, below),
   and it is never meant to be edited in place. Create your own copy instead:
   ```
   cp config/default.toml config/live.toml
   ```
   Then open `config/live.toml` and change `live_api_enabled = false` to
   `live_api_enabled = true` under `[execution]`. This file is already in
   `.gitignore` -- it is yours alone, never committed/pushed, and every
   participant (and the repo itself) creates it independently after
   cloning. **If you skip this step, every `--live` command below will
   fail** with:
   ```
   lab_core.lab_cli_support.LiveExecutionNotAllowed: Live execution requested
   (--live) but execution.live_api_enabled is false in configuration.
   ```
   This is not a bug -- it is the exact safety check working as designed.
   The fix is always the same: create `config/live.toml` as above, and add
   `--config config/live.toml` to the command.
4. If you're on the Docker path, rebuild the image once after creating
   `config/live.toml` so it's copied into the container:
   ```
   docker compose build
   ```
5. Build and run: see **Running with Docker** or **Running without Docker**
   below. Every `--live` example from here on assumes `config/live.toml`
   exists; the commands include `--config config/live.toml` explicitly.

## Running with Docker (recommended)

You don't need Python or `uv` installed at all -- only Docker.

```
# 1. Build the image (first time, or after pulling repo changes)
docker compose build

# 2. Smoke-test: should print the CLI help with all 9 labs listed
docker compose run --rm lab-runner --help
```

If step 2 prints the help text with `lab01` through `lab09` listed, the
setup is working. Next, run a lab end to end without needing an API key
yet, to confirm results actually come back out to your machine:

```
# Lab07 is fully simulated -- no API key, no --live, needed.
docker compose run --rm lab-runner lab07 all

# Check the results landed on your HOST machine (results/ is bind-mounted)
cat results/lab07_rate_limiting.json
```

For any other lab, which does call the real Groq API, pass `--live` and
`--config config/live.toml` (see Setup, step 3), and make sure
`GROQ_API_KEY` is set in your shell (Compose reads it from the environment
via `environment: - GROQ_API_KEY=${GROQ_API_KEY}` in `compose.yaml`):

```
docker compose run --rm lab-runner lab01 short_prompt --live --config config/live.toml
```

### Validating the Docker setup itself (optional, for troubleshooting)

To confirm the image builds cleanly and the full offline test suite (71
tests, no API key or network needed) passes *inside the container*:

```
docker compose build
docker compose run --rm --entrypoint uv lab-runner run pytest -q
```

If this doesn't end with `... passed`, something about your local Docker
build is broken independently of any lab or API key -- fix this first
before troubleshooting a specific lab.

## Running without Docker (uv)

```
uv sync
uv run lab-cli --help
uv sync --group dev && uv run pytest -q   # optional: run the offline test suite
```

## Safety: when does a lab actually call the Groq API?

Every lab (except Lab07, which is fully simulated and never calls the real
API) requires **both**:

1. `execution.live_api_enabled = true` in the config file you pass (your own
   `config/live.toml` from Setup step 3 -- `config/default.toml` always
   defaults this to `false`), **and**
2. the `--live` flag on the command line.

Without both, a lab prints an explanation and exits without sending any
request or incurring any cost. This is deliberate -- running a lab with
`--live` sends real requests to the Groq API using your own key and may
incur real, if typically very small, cost on your account.

General pattern: `uv run lab-cli <lab_id> <experiment> --live --config config/live.toml [options]`,
or the Docker equivalent: `docker compose run --rm lab-runner <lab_id> <experiment> --live --config config/live.toml [options]`.
Every example below uses the `uv run lab-cli ...` form; substitute
`docker compose run --rm lab-runner ...` if you're on the Docker path.
Results are written as both JSON and CSV under `results/`.

**Model availability can change.** Groq's hosted model list is not fixed --
during this project's own live verification, two models it originally
shipped with (`llama-3.1-8b-instant`, `llama-3.3-70b-versatile`) had already
been removed from the account, producing `groq.NotFoundError: ...
model_not_found`. If you hit this, check which models your key can actually
use and update `[groq] default_model` (and `[lab05]`'s model pair) in your
`config/live.toml` accordingly:
```
uv run python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data]"
# or, via Docker:
docker compose run --rm --entrypoint uv lab-runner run python -c "from groq import Groq; [print(m.id) for m in Groq().models.list().data]"
```

## Lab01 -- Token Anatomy & Cost Calculation

**Objective:** see the input/output token breakdown Groq reports for a
request, and how estimated cost is computed from it.

```
uv run lab-cli lab01 all --live --config config/live.toml
```

Compare `short_prompt` vs `long_prompt`: prompt_tokens should scale with
prompt length; completion_tokens depends on what the model chooses to say.
Check `results/lab01_token_anatomy.csv` for the full breakdown and
`estimated_cost_usd` per call.

The terminal output also breaks down, per run, `reasoning_tokens` and the
derived `non_reasoning_tokens` (= completion_tokens - reasoning_tokens, only
when the provider reports reasoning_tokens) and `finish_reason` ("stop" vs
"length" -- the latter means generation was cut off at a token limit, not a
natural stop). After all repetitions of an experiment, a summary block
reports min/max/mean/standard deviation for completion tokens (and
reasoning tokens, when available) and is also written to
`results/lab01_token_anatomy_summary.json`. With the default 5 repetitions,
treat that summary as descriptive of one run, not a statistically robust
benchmark.

## Lab02 -- Context Growth & Context Bloat

**Objective:** observe how growing or padding the context before a
question changes prompt_tokens, cost, and latency -- while the question
and its correct answer stay the same.

```
uv run lab-cli lab02 all --live --config config/live.toml --turns 5
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
uv run lab-cli lab03 all --live --config config/live.toml
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
uv run lab-cli lab04 all --live --config config/live.toml
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
uv run lab-cli lab05 all --live --config config/live.toml
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
uv run lab-cli lab06 all --live --config config/live.toml
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
uv run lab-cli lab08 all --live --config config/live.toml --max-concurrent 4
```

Compare total wall-clock time, `requests_per_second`, and p50/p95 latency
between `sequential` and `concurrent`. Concurrency should reduce total time
but not necessarily per-call latency.

## Lab09 -- Batch Processing

**Objective:** contrast three distinct concepts: synchronous requests,
Python-level async/concurrent requests, and the provider-managed Groq
Batch API.

```
uv run lab-cli lab09 sequential --live --config config/live.toml
uv run lab-cli lab09 async --live --config config/live.toml
uv run lab-cli lab09 prepare_batch_input --config config/live.toml   # no --live needed, pure local file write
uv run lab-cli lab09 submit_batch --live --config config/live.toml   # submits only; does NOT wait
uv run lab-cli lab09 check_batch_status --live --config config/live.toml --batch-id <id>
uv run lab-cli lab09 fetch_batch_results --live --config config/live.toml --output-file-id <id>
```

**Note:** on at least one real Groq account, `submit_batch` returned a real
`403 Permission Denied: Not available for your plan` -- the Batch API is
not enabled for every plan tier. If you hit this, it's an account
limitation, not a bug in this lab; ask your instructor for a sample
captured result to discuss instead.

**A real batch job's completion window is 24 hours to 7 days** -- it will
not finish during a class session. See
`docs/instructor/lab09_sample_batch_result.json` for a pre-captured sample
to discuss the Batch API's output shape and cost trade-off without waiting.

## Troubleshooting

- `GROQ_API_KEY is not set`: export it (`.env` via Docker Compose, or your
  shell environment when using `uv` directly).
- `Live execution requested but execution.live_api_enabled is false`: you
  haven't created `config/live.toml` yet (Setup, step 3), or forgot
  `--config config/live.toml` on this specific command.
- `model_not_found` (404): the model in your config no longer exists on
  Groq's side -- see the "Model availability can change" note under
  **Safety**, above.
- `403 Permission Denied: Not available for your plan`: the API feature
  you're calling (e.g. the Batch API in Lab09) isn't enabled for your
  account's plan tier -- not a bug, a real access restriction.
- Results look identical across repetitions: real models and networks are
  non-deterministic; run with more `--repetitions` before drawing conclusions
  from a single call.
- `docker: error during connect` / `Cannot connect to the Docker daemon`:
  Docker Desktop (or the Docker Engine service) isn't running -- start it
  and wait for it to report "running" before retrying.
- Docker build fails on `uv sync` with a missing file error: make sure
  you're building from the repository root (`docker compose build` run
  from the directory containing `compose.yaml`), not a subdirectory.
- Not sure if the Docker setup itself (vs. a specific lab or API key) is
  the problem? Run the in-container test suite first:
  `docker compose run --rm --entrypoint uv lab-runner run pytest -q` --
  if that passes, the image and CLI wiring are fine and the issue is
  specific to the lab/API key you were testing.
