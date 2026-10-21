# Token Economics, Cost & API Performance Labs

A hands-on training project covering token economics, API cost calculation,
and API performance (latency, caching, batching, concurrency) using real
requests to the Groq API.

## Status

**Stage 0 complete: project foundation only.** No labs are implemented yet.
This repository is built incrementally, one lab at a time, each going
through its own design/review/approval/implementation/testing cycle.

## Roadmap

| Stage | Lab | Status |
|---|---|---|
| 0 | Project foundation (config, shared measurement infra, Docker, tests) | Done |
| 1 | Lab01 — Token Anatomy & Cost Calculation | Not started |
| 2 | Lab02 — Context Growth & Context Bloat | Not started |
| 3 | Lab03 — Prompt Caching | Not started |
| 4 | Lab04 — Output Control & Invocation Optimization | Not started |
| 5 | Lab05 — Model Routing | Not started |
| 6 | Lab06 — Streaming & Perceived Latency | Not started |
| 7 | Lab07 — Rate Limiting & Retry Strategies | Not started |
| 8 | Lab08 — Async Processing & Concurrency | Not started |
| 9 | Lab09 — Batch Processing | Not started |

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
- Instructors: `docs/instructor/README.md`

## Development

```
uv sync --group dev
uv run pytest            # offline unit tests only (no API key needed)
```

Tests marked `@pytest.mark.integration` require a real `GROQ_API_KEY` and
are excluded by default.
