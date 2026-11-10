# LLM Token Economics Lab

**A reproducible Python lab for understanding LLM token consumption, estimated API costs, and performance trade-offs.**

Explore 10 implemented experiments covering token accounting, context growth, caching, output control, model routing, streaming, rate limiting, concurrency, batching, and summarization. The repository includes a shared Python CLI, offline tests, GitHub Actions quality checks, and a **read-only Streamlit dashboard**.

**Live API provider:** Groq. **Lab07:** offline simulation with no provider requests.

> **Safe by default.** Running the dashboard, tests, or Lab07 does not send requests to Groq. Other labs require **both** `--live` on the CLI and `live_api_enabled = true` in a local config. Live calls can incur cost.

## Quick start — Docker (recommended)

Requirements: [Docker Desktop](https://www.docker.com/products/docker-desktop/) with Linux containers on Windows, or Docker Engine with the Compose plugin on Linux/macOS. **No local Python installation or API key is needed for this example.**

Open PowerShell or a terminal **in the repository root** (the folder containing `compose.yaml`).

```powershell
# Build both the existing CLI image and the Streamlit image.
docker compose build lab-runner dashboard

# Start the dashboard in the background.
docker compose up -d dashboard

# Create an API-free Lab07 simulation and save results to ./results.
docker compose run --rm lab-runner lab07 all

# Confirm the dashboard service is running / healthy.
docker compose ps
```

Open **http://localhost:8501**, click **Refresh results**, and select the **Retry simulator** tab.

To see the container logs or stop the dashboard:

```powershell
docker compose logs dashboard
docker compose stop dashboard
```

The dashboard is deliberately bound to **127.0.0.1:8501**, not the public network. Its container runs as a non-root user, has a read-only filesystem, and mounts `./results` **read-only**. The separate `lab-runner` service can write experiment files to that directory. The dashboard receives **no Groq API key** and cannot launch provider experiments.

After updating the repository, rebuild the images to pick up changes:

```powershell
git pull origin master
docker compose build lab-runner dashboard
docker compose up -d dashboard
```

## Quick start — Python + uv

Requires **Python 3.12+** and [uv](https://docs.astral.sh/uv/). Run the following from the repository root:

```powershell
uv sync --frozen --group dev
uv run lab-cli --help
uv run lab-cli lab07 all
uv run --with "streamlit>=1.40,<2" streamlit run dashboard/app.py
```

Open **http://localhost:8501** and click **Refresh results**. Keep the Streamlit terminal running while using the dashboard; stop it with **Ctrl+C**.

Choose **one** launch method for the dashboard. If it is already running locally, stop that process before starting the Docker service, because both use port 8501 by default.

## The 10 laboratories

| Lab | Experiment | Type |
| --- | --- | --- |
| **Lab01** | Token anatomy, completion/reasoning tokens, estimated cost | Live Groq, explicit opt-in |
| **Lab02** | Context growth and context bloat | Live Groq, explicit opt-in |
| **Lab03** | Prompt caching and estimated savings | Live Groq, explicit opt-in |
| **Lab04** | Output length and invocation optimization | Live Groq, explicit opt-in |
| **Lab05** | Model routing and cost/quality trade-offs | Live Groq, explicit opt-in |
| **Lab06** | Streaming and perceived latency | Live Groq, explicit opt-in |
| **Lab07** | Rate limits, retries, exponential backoff and jitter | **Offline simulation** |
| **Lab08** | Asynchronous execution and concurrency | Live Groq, explicit opt-in |
| **Lab09** | Batch processing | Live Groq, explicit opt-in |
| **Lab10** | Conversation summarization and context economics | Live Groq, explicit opt-in |

Find the available experiments and flags using:

```powershell
uv run lab-cli --help
uv run lab-cli lab07 --help
```

For detailed per-lab setup and run instructions, see the [Participant Guide](docs/README.md).

## Dashboard and saved results

The dashboard **reads existing local files**; it does not automatically execute laboratories, fetch missing experiments from GitHub, or fabricate metrics.

| Dashboard tab | What it shows |
| --- | --- |
| **Overview** | Discovered runs, measurement counts and coverage |
| **Model comparison** | Recorded cost, token and latency comparisons within one lab and run |
| **Retry simulator** | Lab07 strategy comparisons within one run and scenario |
| **Measurements** | Allowlisted, normalized metrics and CSV export |

**Why are the other labs marked “No saved results”?** All ten appear in the **Lab** filter even when they have never been run. Only labs with measurement files under your local `results/` folder can display actual experiment charts. If you run just Lab07, it is normal for Lab01–06 and Lab08–10 to have no recorded results.

Lab07 saves each run separately, for example:

```text
results/
└── lab07_rate_limiting/
    └── <run_id>/
        ├── experiment_manifest.json
        ├── raw_measurements.json
        ├── raw_measurements.csv
        ├── strategy_summary.csv
        └── analysis_report.md
```

To inspect saved runs on Windows:

```powershell
Get-ChildItem .\results\lab07_rate_limiting -Recurse
```

`results/` is ignored by Git: **runs on different computers are not synchronized automatically**. Data collected by one contributor will not appear in another contributor's dashboard until its supported result files are made available locally.

**Interpreting the metrics:** API costs are configured **estimates**, not invoices. Missing metrics remain missing rather than being reported as zero. Lab07 success rates refer to simulated HTTP/client outcomes; they are **not model answer accuracy**, and its virtual time is **not real request latency**. Different labs and unmatched workloads should not be compared as if they were equivalent.

Read the [Dashboard Guide](docs/dashboard.md) for additional detail.

## Running live Groq experiments

**Skip this section if you only want the dashboard, offline tests or Lab07.**

Live Groq calls require an API key and two deliberate opt-ins. Only proceed if you intend to make real requests.

1. Copy `config/default.toml` to `config/live.toml` and edit the `[execution]` section to set `live_api_enabled = true`.
2. Provide your personal `GROQ_API_KEY` through your shell environment (uv) or `.env` (Docker Compose).
3. Supply **both** `--live` and `--config config/live.toml` for the requested lab.

**PowerShell with uv:**

```powershell
Copy-Item config/default.toml config/live.toml
# Edit config/live.toml to enable live calls.
# Set GROQ_API_KEY in your PowerShell environment separately.
uv run lab-cli lab01 short_prompt --live --config config/live.toml
```

The uv CLI reads `GROQ_API_KEY` from the **process environment**. **Creating a `.env` file by itself does not load the key into a local uv command.** Check that the variable is set without printing its value, and avoid putting real keys in command history.

**Docker Compose:**

```powershell
Copy-Item .env.example .env
Copy-Item config/default.toml config/live.toml
# Put GROQ_API_KEY in .env; enable live calls in config/live.toml.
docker compose build lab-runner
docker compose --env-file .env run --rm lab-runner lab01 short_prompt --live --config config/live.toml
```

The CLI Dockerfile copies `config/` at build time, so **rebuild the CLI image after creating or changing** `config/live.toml`. Never store the API key in this config file: use the environment or a local `.env` instead. Do not commit `.env`, `config/live.toml`, or private experiment outputs.

Model availability and API pricing may change. Confirm your account's accessible models and the provider's current pricing before live runs. For the exact experiment modes, use `uv run lab-cli <lab_id> --help`.

## Tests, CI, and security

```powershell
uv sync --frozen --group dev
uv run pytest -q
```

You can also run tests inside the CLI image:

```powershell
docker compose build lab-runner
docker compose run --rm --entrypoint uv lab-runner run pytest -q
```

GitHub Actions checks include the offline pytest suite, the **Docker dashboard smoke test** (build, health, simulated data visibility, and API-key isolation), **Ruff** linting, targeted **mypy** checks, test coverage (initial 65% floor), and **Bandit** security analysis. The dependency audit is initially **advisory**; a green workflow is not a complete security certification.

The Docker dashboard smoke test starts the service without live API calls. The dashboard is intended for **trusted local use**, not unauthenticated public hosting.

See [CONTRIBUTING.md](CONTRIBUTING.md) for contribution standards and [SECURITY.md](SECURITY.md) for private vulnerability reporting.

## Troubleshooting

| Problem | What to check |
| --- | --- |
| Port 8501 already in use | Stop the other Streamlit session or dashboard container; do not run both launch methods simultaneously. |
| Docker dashboard not opening | Check `docker compose ps` and `docker compose logs dashboard`; confirm Docker Desktop is running Linux containers. |
| Lab filter lists only Lab07 | Pull the latest `master` and restart Streamlit. All ten lab names should be selectable. |
| Lab listed but graphs are empty | The lab has no supported locally saved measurements yet; select a run with data or execute the intended lab separately. |
| `N/A` on Lab07 cost, latency or tokens | Expected: Lab07 is a virtual retry simulator and makes zero real API requests. |
| `ImportError` after updating | Stop the old Streamlit process, update the correct checkout, and start the dashboard again. |
| Live call says key is missing | Check `GROQ_API_KEY` in the process environment for uv, or `.env` for Docker Compose. |
| Live model returns `model_not_found` | Review model access for your Groq account and update `config/live.toml`. |

## Collaboration and project status

The development fork is [ipekgulsencelik/llm-token-economics-lab](https://github.com/ipekgulsencelik/llm-token-economics-lab); the upstream project is [Burakkylmz/llm-token-economics-lab](https://github.com/Burakkylmz/llm-token-economics-lab). Changes are proposed as Pull Requests and tested before being merged. To update a local checkout, pull from whichever repository your `origin` remote points to; use `git remote -v` to check.

The repository currently has **no published software license**. The maintainers must select and add one before representing the project as open-source licensed.
