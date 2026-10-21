# Token Economics, Cost & API Performance Labs

A hands-on training project covering token economics, API cost calculation,
and API performance (latency, caching, batching, concurrency) using real
requests to the Groq API.

## Status

**All 9 labs implemented, on top of the shared Stage 0 foundation.**
Every lab's logic is covered by fully offline unit tests (mocked Groq
client; no `GROQ_API_KEY` or network needed to run the test suite). Real
Groq API calls only ever happen when a participant explicitly passes
`--live` with `execution.live_api_enabled = true` (Lab07 is the one
exception: it never calls the real API at all, by design).

## Roadmap

| Stage | Lab | Status |
|---|---|---|
| 0 | Project foundation (config, shared measurement infra, Docker, tests) | Done |
| 1 | Lab01 — Token Anatomy & Cost Calculation | Done |
| 2 | Lab02 — Context Growth & Context Bloat | Done |
| 3 | Lab03 — Prompt Caching | Done |
| 4 | Lab04 — Output Control & Invocation Optimization | Done |
| 5 | Lab05 — Model Routing | Done |
| 6 | Lab06 — Streaming & Perceived Latency | Done |
| 7 | Lab07 — Rate Limiting & Retry Strategies | Done |
| 8 | Lab08 — Async Processing & Concurrency | Done |
| 9 | Lab09 — Batch Processing | Done |

Run `uv run lab-cli --help` to see all registered labs, or
`uv run lab-cli <lab_id> --help` for a specific lab's options. See
`docs/participant/README.md` for per-lab run commands and what to observe.

## Requirements

- Docker Engine/Desktop + Docker Compose (recommended path), **or**
- [`uv`](https://docs.astral.sh/uv/) if you prefer to run without Docker.
- A personal Groq API key (`GROQ_API_KEY`).

## Setup

1. Copy `.env.example` to `.env` and set your own `GROQ_API_KEY`.
2. Without Docker: `uv sync`, then `uv run lab-cli --help`.
3. With Docker: `docker compose run --rm lab-runner --help`.

No real Groq API call is ever made without an explicit `--live` flag at
the CLI, on top of `execution.live_api_enabled = true` in configuration
(see `config/default.toml`). This is a deliberate safety default.

## Documentation

- Participants: `docs/participant/README.md`
- Instructors: `docs/instructor/README.md` (includes
  `lab09_sample_batch_result.json`, a pre-captured sample for Lab09 since a
  real Groq Batch API job can take 24h-7d to complete)

## Development

```
uv sync --group dev
uv run pytest            # offline unit tests only (no API key needed)
```

Tests marked `@pytest.mark.integration` require a real `GROQ_API_KEY` and
are excluded by default.
