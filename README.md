# LLM Token Economics Lab

A hands-on Python laboratory for exploring **token usage, estimated LLM API
cost, latency, context growth, prompt caching, routing, streaming, retries,
concurrency, batching, and conversation summarization**.

It contains **10 implemented labs**, a shared experiment infrastructure,
automated offline tests, Docker support, and an optional **Streamlit
dashboard** for exploring saved results. Groq is the live API provider used
by the experiments; Lab07 uses a local simulator only.

> **Cost safety:** Live provider calls are disabled by default. Every real
> Groq experiment requires **both** the CLI flag `--live` **and**
> `execution.live_api_enabled = true` in a local config file. The
> dashboard never initiates provider requests.

## Quick start: no API key required

Requires Python **3.12+** and [uv](https://docs.astral.sh/uv/), or Docker
Desktop/Engine with Compose.

```powershell
# From the repository root (PowerShell, macOS or Linux terminal):
uv sync --group dev
uv run lab-cli --help

# Run an entirely offline retry / rate-limit simulation:
uv run lab-cli lab07 all

# Run all offline tests:
uv run pytest -q

# Open the read-only visual dashboard:
uv run --with "streamlit>=1.40,<2" streamlit run dashboard/app.py
```

Streamlit displays its local address in the terminal (normally
[http://localhost:8501](http://localhost:8501)). In the dashboard, select
**Retry simulator** to compare Lab07 strategies. Select a run and a
scenario to see success rate, retry count, simulated HTTP 429 responses,
and virtual wait time. The dashboard is **local-only** and reads existing
files under `results/`.

The **Lab** selector always shows all ten labs. `No saved results` means
that the relevant lab has not yet written measurements to your local
`results/` folder; it does **not** mean its implementation is missing.

## Experiment catalog

| Lab | Experiment | Execution |
|---|---|---|
| 01 | Token anatomy, cost breakdown, reasoning effort | Live API with explicit opt-in |
| 02 | Context growth and context bloat | Live API with explicit opt-in |
| 03 | Prompt caching and cache economics | Live API with explicit opt-in |
| 04 | Output control and invocation optimization | Live API with explicit opt-in |
| 05 | Model routing and quality/cost trade-offs | Live API with explicit opt-in |
| 06 | Streaming and perceived latency | Live API with explicit opt-in |
| 07 | Rate limits and retry strategies | **Offline simulation** |
| 08 | Asynchronous execution and concurrency | Live API with explicit opt-in |
| 09 | Batch processing and asynchronous workflows | Live API with explicit opt-in |
| 10 | Conversation summarization and context economics | Live API with explicit opt-in |

See [the participant guide](docs/README.md) for per-lab commands and
experiment design, and [the dashboard guide](docs/dashboard.md) for how to
interpret the graphs.

## Running with Docker

Docker can run the **CLI experiments and tests** without installing Python
or uv on the host. The Streamlit dashboard is currently launched with
**uv on the host**, not through the `lab-runner` container.

```bash
docker compose build
docker compose run --rm lab-runner --help
docker compose run --rm lab-runner lab07 all
```

Lab07 saves files in a per-run directory:

```text
results/
  lab07_rate_limiting/
    <experiment_run_id>/
      raw_measurements.json
      raw_measurements.csv
      strategy_summary.csv
      analysis_report.md
```

The `results/` directory is bind-mounted to the host by `compose.yaml`.
Use your file explorer or list the folder from a terminal:

```powershell
Get-ChildItem results/lab07_rate_limiting -Recurse
```

Check the offline test suite inside Docker:

```bash
docker compose run --rm --entrypoint uv lab-runner run pytest -q
```

No Groq key is needed for Lab07 or the offline tests.

## Live experiments: explicit opt-in

Only follow these steps if you intend to make **real requests** and accept
potential API charges.

1. Copy `.env.example` to `.env` and supply your own `GROQ_API_KEY`.
2. Copy `config/default.toml` to `config/live.toml` (a git-ignored
   personal config).
3. In `config/live.toml`, change `live_api_enabled = false` to
   `live_api_enabled = true` under `[execution]`.
4. Invoke your chosen experiment with **both** `--live` and
   `--config config/live.toml`. For example:

```powershell
Copy-Item .env.example .env
Copy-Item config/default.toml config/live.toml
# Edit .env and config/live.toml in your editor, then explicitly run:
uv run lab-cli lab01 short_prompt --live --config config/live.toml
```

For Docker-based live execution, configure the environment and rebuild
after creating `config/live.toml` (the Dockerfile copies `config/`
at build time):

```bash
docker compose build
docker compose --env-file .env run --rm lab-runner lab01 short_prompt --live --config config/live.toml
```

Do not commit keys, `.env`, or `config/live.toml`. Model availability
and provider prices can change; review your accessible models and pricing
before running experiments. Project cost values are **estimates**, not
billing records. Without the required explicit live flags/configuration,
no live Groq requests are made.

## Dashboard: interpreting results

The dashboard is a **read-only** analytics viewer:

- **Overview:** saved experiment runs, request counts and coverage.
- **Model comparison:** descriptive cost, latency and token comparisons
  within the **same lab and run**.
- **Retry simulator:** four retry-strategy charts for **one Lab07 run and
  one simulation scenario** at a time.
- **Measurements:** safe, normalized per-request metrics and CSV export.

Missing metrics show `N/A` rather than fabricated zeros. In Lab07,
actual provider token usage, billed cost and model-answer accuracy are
unavailable **by design**; the success percentage indicates simulated
client success, **not** LLM response correctness. Other labs require
saved experiment measurements before their plots can appear.

The dashboard loads approved measurement files only; it does not export
raw prompts, model responses or conversation traces.

## Development and CI

```bash
uv sync --frozen --group dev
uv run pytest -q
```

GitHub Actions runs offline pytest on pushes and pull requests and
syntax-checks the dashboard entry point. Tests marked
`@pytest.mark.integration` are excluded by default and require
separately authorized live access.

## Collaboration and upstream repository

This repository is developed in collaboration using GitHub forks and
Pull Requests:

- [Burakkylmz/llm-token-economics-lab](https://github.com/Burakkylmz/llm-token-economics-lab) — upstream project.
- [ipekgulsencelik/llm-token-economics-lab](https://github.com/ipekgulsencelik/llm-token-economics-lab) — development fork.

Changes should be reviewed and tested before merging into the upstream
`master` branch. The same dashboard and safety rules apply in both
repositories once those changes are merged.

## Troubleshooting

- **Only Lab07 is listed in saved results:** update to the current version
  and restart Streamlit. The Lab selector lists all ten labs, while only
  recorded experiments have measurements.
- **A dashboard import fails after updating:** stop Streamlit, run
  `git pull origin master` in your project checkout, and restart it.
- **`N/A` for cost/latency on Lab07:** expected, because it uses a
  virtual simulator, not real Groq calls.
- **`model_not_found` during a live experiment:** verify which Groq models
  your own key permits and update your local live config.
- **A file or run is missing:** confirm you are reading the same local
  `results/` directory. Experiment outputs are git-ignored and are
  not automatically synchronized between contributors.

Instructor-only materials are intentionally not published in this
public repository; do not follow links to private `docs/instructor/`
paths from older instructions.
