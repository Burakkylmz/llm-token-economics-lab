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

The same command also writes PNG charts automatically -- no extra command
needed. Check `results/lab01_token_anatomy/charts/` for `short_prompt_*.png`
/ `long_prompt_*.png` (completion tokens per run, reasoning vs
non-reasoning tokens, finish reasons, and a combined `*_dashboard.png`).
This applies to `short_prompt`, `long_prompt`, and `all` alike.

### Lab01 effort mode -- reasoning_effort comparison

A separate execution mode compares `reasoning_effort` levels only
(low/medium/high) for a fixed prompt, to study their effect on
reasoning/completion token consumption, latency, and cost. (An earlier,
much larger `factorial` mode that also varied `temperature`/`top_p` across
540 calls was retired due to Groq rate-limit/quota pressure; `lab01
factorial` now always fails closed with a message pointing here.) It never
touches `lab01_token_anatomy.{json,csv}`; results go to their own
`results/lab01_effort/<experiment_run_id>/` folder.

```
# Preview the plan (6 calls by default) -- never calls the API, needs no API key:
uv run lab-cli lab01 effort --dry-run --config config/default.toml

# Minimal pilot (3 calls: 1 repetition instead of 2):
uv run lab-cli lab01 effort --dry-run --repetitions 1 --config config/default.toml

# Extended experiment (12 calls: adds long_prompt):
uv run lab-cli lab01 effort --dry-run --prompt short_prompt long_prompt --config config/default.toml

# Real execution (real cost -- see config/default.toml's [lab01.effort]
# section for the request/cost budget caps that apply):
uv run lab-cli lab01 effort --live --config config/live.toml
```

A successful live run automatically writes 6 PNG charts to
`results/lab01_effort/<run_id>/charts/` (reasoning vs non-reasoning tokens,
completion tokens/latency/cost/reasoning-ratio by effort, and a combined
summary dashboard). To (re)generate charts for any already-completed run --
including the retired `factorial` mode's old results -- without calling the
API at all:

```
# Regenerate charts for every completed lab01_effort/lab01_factorial run:
uv run lab-cli lab01 backfill
# or, in Docker:
docker compose run --rm lab-runner lab01 backfill
```

See `docs/instructor/README.md`'s "LAB01 -- Reasoning Effort Comparison"
section for the full experimental design, statistics methodology, and
troubleshooting guide (ask your instructor if you don't have access to
that file).

## Lab02 -- Context Growth & Context Bloat

**Objective:** observe how growing or padding the context before a
question changes prompt_tokens, reasoning tokens, cost, latency, and
answer correctness -- while the question, expected answer, and system
instruction stay identical across conditions.

This project's documented default plans exactly 10 API calls: 3 context
conditions (`minimal_context`, `expanded_context`, `irrelevant_context`) x
2 repetitions, plus 4 `multi_turn_growth` turns.

Preview the plan first, with no API key and no cost, via `--dry-run`:

```
uv run lab-cli lab02 all --dry-run --config config/default.toml
```

Then run it for real:

```
uv run lab-cli lab02 all --live --config config/live.toml
```

Each run writes into its own isolated results folder --
`results/lab02_context_growth/<experiment_run_id>/` -- containing
`context_measurements.json/.csv`, `multi_turn_measurements.json/.csv`,
`condition_summary.csv`, `experiment_summary.json`, `analysis_report.md`,
and PNG charts in `charts/`, written automatically in the same command
(no extra step). A CLI override that would push the total above
`lab02.max_api_calls` (10 by default) is refused up front, before any API
call is made -- lower `--repetitions`/`--turns`, or raise
`lab02.max_api_calls` deliberately in your own config.

Compare `minimal_context`, `expanded_context`, and `irrelevant_context` in
`charts/context_size_vs_latency.png` / `token_breakdown.png` /
`reasoning_tokens_by_context.png` / `estimated_cost_by_context.png` /
`answer_correctness.png`: `prompt_tokens`, reasoning-token usage, and
timing should move with context size/relevance, while
`answer_correctness.png` should show every condition still answering
correctly most of the time, since the relevant fact is present in every
condition (answer correctness is graded by a strict exact-integer check,
not a substring match -- see `docs/instructor/README.md`). Then look at
`multi_turn_growth.png` / `cumulative_token_consumption.png`:
`prompt_tokens` increases turn over turn, and the *cumulative* input-token
total across the whole conversation grows even faster, since every prior
turn's reply is re-sent as input on every later turn.

To regenerate charts for existing Lab02 results (both new per-run folders
and the pre-redesign flat `results/lab02_context_growth.json` file)
without making any API call:

```
uv run lab-cli lab02 backfill
# or, in Docker:
docker compose run --rm lab-runner lab02 backfill
```

See `docs/instructor/README.md`'s "LAB02 -- Context Growth & Context
Bloat" section for the full experimental design, the answer-correctness
evaluator, and documented confounders (ask your instructor if you don't
have access to that file).

## Lab03 -- Prompt Caching

**Objective:** observe Groq's automatic prompt caching (currently
`openai/gpt-oss-20b`, `openai/gpt-oss-120b`, `openai/gpt-oss-safeguard-20b`
only), measure its effect on cost and latency, and distinguish
"cache-eligible" from "an actually observed cache hit".

The default experiment is a small, paired/interleaved 6-call plan: 3
repetitions x 2 conditions (`repeated_prefix`, `cache_unfriendly`), run as
pair 1 (`repeated_prefix` then `cache_unfriendly`), pair 2, pair 3 -- never
all of one condition followed by all of the other. Both conditions in a
pair ask the SAME paired question (one of 4 built-in questions, each with
its own deterministic integer answer drawn from a realistic customer-
support knowledge base) and have comparable prompt length; the only thing
that systematically differs is whether the leading prefix can repeat
byte-for-byte across calls.

```
uv run lab-cli lab03 all --dry-run --config config/live.toml
uv run lab-cli lab03 all --live --config config/live.toml
```

`--repetitions 2` gives a 4-call minimal pilot; `--repetitions 4` gives an
8-call extended run (this project's documented ceiling,
`lab03.max_api_calls`). You can also run only one condition:
`uv run lab-cli lab03 repeated_prefix --live --config config/live.toml`.

**`cache_probe` -- the strongest exact-match condition.** If `lab03 all`
never shows a cache hit, run the dedicated probe mode first: it sends 3
(default, `lab03.cache_probe_calls`) sequential, BYTE-IDENTICAL requests
(same system tag, same knowledge base, same exact user question -- not
just a shared prefix) -- the single strongest condition this project can
construct for observing `cached_tokens > 0`.

```
uv run lab-cli lab03 cache_probe --dry-run --config config/live.toml
uv run lab-cli lab03 cache_probe --live --config config/live.toml
```

The dry-run prints an offline "Prefix Stability Verification" block
(hash + lengths) proving the 3 planned requests really are identical,
before any API call. The live run ends with a `Cache Verification:
OBSERVED`/`NOT OBSERVED` line and a concrete, evidence-based reason --
never a bare pass/fail.

**Results and charts.** Every live run writes to its own
`results/lab03_prompt_caching/<experiment_run_id>/` directory
(`cache_measurements.json`/`.csv`, `cache_diagnostics.json`,
`condition_summary.csv`, `experiment_summary.json`, `analysis_report.md`)
and never touches or overwrites the older flat
`results/lab03_prompt_caching.json`/`.csv` files. `cache_diagnostics.json`
is a small, bounded, safe projection (provider model, SDK version,
request number, token counts, cache status, raw usage field names) --
never prompt or completion text.

Charts are generated automatically -- no extra command needed -- into
`<run_id>/charts/`: cached vs. uncached input tokens, cache hit/miss/
unavailable by request, token cache ratio by request, a cache-friendly-
vs-unfriendly comparison, estimated cost by condition, estimated cost
savings, latency by condition, cached tokens vs. latency, a summary
dashboard, and (for `cache_probe` runs only) an exact-match probe
sequence chart.

**Two different cache metrics, shown separately, never confused:**
"observed request cache hit rate" (what fraction of REQUESTS had any
cache hit) and "token cache ratio" (what fraction of all prompt TOKENS,
across reporting requests, were cached) answer different questions and
are not interchangeable.

**A cache miss on `repeated_prefix` is not necessarily a bug** -- caching
is provider-managed, has an approximate multi-hour TTL, and is never
guaranteed. Every call is classified into one of 4 statuses
(`CACHE_HIT` / `CACHE_MISS` / `CACHE_USAGE_UNAVAILABLE` / `INVALID_USAGE`),
printed as `Cache Status:` on every call and shown in its own
"Cache Hit / Miss / Unavailable by Request" chart.
`CACHE_USAGE_UNAVAILABLE` (the provider never reported cache usage for
that call at all) is NEVER the same thing as `CACHE_MISS` (the provider
reported `cached_tokens == 0`) -- if you only ever see
`CACHE_USAGE_UNAVAILABLE`, that is not an SDK bug on this project's side
(see the instructor guide's root-cause section for how that was verified).

Use `uv run lab-cli lab03 backfill --config config/live.toml` to
regenerate charts for any existing run directory, or from the legacy flat
file, without making any API call (no `GROQ_API_KEY` needed). The legacy
backfill never produces an answer-correctness chart, since the old flat
file never stored response text to evaluate.

## Lab04 -- Output Control & Invocation Optimization

**Objective:** see how `max_completion_tokens` limits completion
length/cost -- and whether a restrictive budget damages answer
completeness/correctness -- and how batching several questions into one
call compares to one call per question (token/cost AND workflow-latency).

```
# Preview the plan (10 calls by default: 3 output-limit values x 2
# repetitions + 1 single-invocation call + 3 multiple-invocation calls) --
# never calls the API, needs no API key:
uv run lab-cli lab04 all --dry-run --config config/default.toml

# Real execution:
uv run lab-cli lab04 all --live --config config/live.toml

# or, individually:
uv run lab-cli lab04 output_limit_sweep --live --config config/live.toml
uv run lab-cli lab04 invocation_comparison --live --config config/live.toml

# Regenerate charts for every completed run (and the legacy flat-file
# backfill), offline, no API key:
uv run lab-cli lab04 backfill --config config/default.toml
```

Results land in `results/lab04_output_control/<run_id>/`: measurements
(`output_limit_measurements.json/csv`, `invocation_measurements.json/csv`),
summaries (`condition_summary.csv`, `invocation_comparison.csv`), a
Markdown `analysis_report.md`, and 9 PNG charts under `charts/`.

For `output_limit_sweep`, each condition sends the exact same
retry-policy-extraction task (5 required fields); only
`max_completion_tokens` varies. Truncation is read from the provider's own
`finish_reason` field (`output_truncated` in the results), never guessed
from `completion_tokens == max_completion_tokens` alone. Answer
**completeness** (how many of the 5 fields are present) and **correctness**
(how many present fields have the right value) are reported separately --
a condition that uses fewer tokens but accepts fewer answers is not a
successful optimization.

For `invocation_comparison`, compare the printed `Single Invocation` vs
`Multiple Invocations` summaries: the `Workflow Wall-Clock` line is the
whole strategy's elapsed time (not the sum of individual request
latencies, which are also printed separately), and `Quality Comparison`
never calls a cheaper strategy "better" unless its answers were all
accepted too.

## Lab05 -- Model Routing

**Objective:** measure how model-routing decisions affect answer quality,
token consumption, estimated cost and latency -- comparing an
Always-Small baseline, an optional Always-Large baseline, and a
deterministic, explainable, rule-based router, over a 13-task golden
dataset with 6 different deterministic evaluator types (no substring-only
checks, no LLM judge).

```
# Preview the plan (12 calls by default: 6 tasks x 2 strategies) --
# shows every routing decision OFFLINE (zero API calls), never calls the
# API, needs no API key:
uv run lab-cli lab05 all --dry-run --config config/default.toml

# Real execution:
uv run lab-cli lab05 all --live --config config/live.toml

# Extended profile (adds the Always-Large baseline -- 18 calls):
uv run lab-cli lab05 all --live --profile extended --config config/live.toml

# Single-strategy commands (backward compatible):
uv run lab-cli lab05 fixed_model_baseline --live --config config/live.toml
uv run lab-cli lab05 rule_based_routing --live --config config/live.toml
uv run lab-cli lab05 always_large --live --config config/live.toml

# Regenerate charts for every completed run (and the legacy flat-file
# backfill), offline, no API key:
uv run lab-cli lab05 backfill --config config/default.toml
```

Results land in `results/lab05_model_routing/<run_id>/`: measurements
(`routing_measurements.json/csv`), per-task baseline-vs-routed comparisons
(`task_comparison.csv`), per-strategy summaries (`strategy_summary.csv`),
a Markdown `analysis_report.md`, and 9 PNG charts under `charts/`.

The router reads ONLY each task's prompt text -- never the dataset's
ground-truth `complexity_label`, never the expected answer. Compare the
printed `-- Model Routing Comparison --` block: `Quality Delta`/task-level
`quality_outcome` (`QUALITY_PRESERVED` / `ROUTING_QUALITY_REGRESSION` /
`ROUTING_QUALITY_IMPROVEMENT` / `BOTH_FAILED`) is always reported
separately from cost, so a cheaper routing result is never presented as
"better" if it also got an answer wrong. If routing turns out to be MORE
expensive than the Always-Small baseline, that is reported honestly as a
"Routing Cost Increase" -- not hidden.

## Lab06 -- Streaming & Perceived Latency

**Objective:** distinguish `first_stream_event_ms` (the first SSE packet)
from `first_visible_content_ms` (the first chunk that actually carries
displayable text) from total latency -- and measure whether streaming's
real benefit (showing useful content to the user earlier) actually shows
up in a paired, controlled comparison against a non-streaming call.

```
# Preview the plan (10 calls by default: 5 paired repetitions x 2
# conditions) -- never calls the API, needs no API key:
uv run lab-cli lab06 all --dry-run --config config/default.toml

# Real execution:
uv run lab-cli lab06 all --live --config config/live.toml

# Single-condition commands (backward compatible):
uv run lab-cli lab06 non_streaming --live --config config/live.toml
uv run lab-cli lab06 streaming --live --config config/live.toml

# Extended profile (8 pairs = 16 calls):
uv run lab-cli lab06 all --live --profile extended --config config/live.toml

# Regenerate charts for every completed run (and the legacy flat-file
# backfill), offline, no API key:
uv run lab-cli lab06 backfill --config config/default.toml
```

Results land in `results/lab06_streaming/<run_id>/`: measurements
(`stream_measurements.json/csv`), the per-chunk event timeline
(`stream_events.csv` -- bounded metadata only, never prompt/reasoning
text), paired comparisons (`paired_comparison.csv`), per-condition
summaries (`condition_summary.csv`), a Markdown `analysis_report.md`, and
9 PNG charts under `charts/`.

`ttft_ms` is always `None` in this redesign -- no verified provider
per-token timing distinct from the first SSE event's own arrival time was
found for Groq (see `labs.lab06_streaming.runner`'s module docstring for
the SDK-source-verified investigation). `first_event_to_visible_gap_ms`
shows the gap between the first event and the first MEANINGFUL visible
content (whitespace-only chunks don't count), which can reflect reasoning
generation, metadata-only events, or transport buffering -- the
measurement alone doesn't prove which.

Streaming token usage is requested via `extra_body={"stream_options":
{"include_usage": true}}` (verified supported by the installed Groq SDK);
when the provider doesn't populate it, `usage_status` stays
`USAGE_UNAVAILABLE` and token/cost fields stay `None` rather than being
estimated from chunk counts. Compare the printed `Perceived Latency Gain`
per pair -- a negative value (streaming's first visible content arrived
LATER than the non-streaming full response) is reported exactly as
measured, never hidden.

## Lab07 -- Rate Limiting & Retry Strategies (fully simulated)

**Objective:** compare four retry strategies (`no_backoff`,
`exponential_backoff`, `exponential_backoff_with_jitter`,
`rate_limit_aware`) across five controlled scenarios -- fixed transient
failures, a server recovery window, multi-client shared contention
(thundering herd), Retry-After header variations, and request/token quota
pressure -- using a deterministic virtual clock. This lab never calls the
real Groq API, in any mode, and never needs `GROQ_API_KEY`.

```
# Default Grand Test (3 scenarios x 4 strategies = 12 simulation runs):
uv run lab-cli lab07 all --config config/default.toml

# Pilot (1 scenario x 4 strategies = 4 runs) / Extended (5 scenarios x 4 = 20 runs):
uv run lab-cli lab07 all --profile pilot --config config/default.toml
uv run lab-cli lab07 all --profile extended --config config/default.toml

# Run only one strategy or one scenario:
uv run lab-cli lab07 exponential_backoff_with_jitter --config config/default.toml
uv run lab-cli lab07 all --scenario shared_contention --config config/default.toml

# Legacy flags still work (force a single-scenario run):
uv run lab-cli lab07 no_backoff --fail-count 4 --config config/default.toml

# Regenerate charts for every completed run (and the legacy flat-file
# backfill), offline:
uv run lab-cli lab07 backfill --config config/default.toml
```

Results land in `results/lab07_rate_limiting/<run_id>/`: the per-attempt
`attempt_timeline.csv`, per-client `raw_measurements.json/csv`,
`strategy_summary.csv`, `scenario_comparison.csv` (pivoted by scenario), a
Markdown `analysis_report.md`, and 9 PNG charts under `charts/`.

Compare `success_rate` (reliability) and `total_scheduled_wait_seconds`/
`total_429_responses` (efficiency) SEPARATELY across strategies -- the
strategy with the lowest wait time is never, by itself, the best
strategy. `retry_count` excludes the initial attempt and never counts a
client's final failed attempt as an extra retry. All figures are
synthetic simulation data (`real_api_calls: 0`, `real_token_usage: null`),
never real Groq telemetry.

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
