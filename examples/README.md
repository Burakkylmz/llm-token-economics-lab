# Reproducible experiment examples

These examples are **offline** and require no Groq API key. They show how to
run Lab07 with a fixed seed, collect an existing saved run, and create
auditable Markdown and JSON benchmark reports.

## PowerShell (Windows 11)

From the repository root:

```powershell
uv sync --frozen --group dev
.\examples\run_lab07_and_report.ps1
```

Or use Docker for the Lab07 run if uv is not installed, then use the
Python/uv reporter on the host once available. The reporter is **not**
embedded in the Streamlit dashboard and does not initiate API calls.

## Bash (Linux / macOS / WSL)

```bash
uv sync --frozen --group dev
bash examples/run_lab07_and_report.sh
```

Each script runs `lab07 all --profile pilot --seed 42` using a unique,
timestamped run ID, then saves two reports under `results/benchmarks/`:

- `<run_id>.md`: human-readable strategy comparison
- `<run_id>.json`: structured numeric summary with per-metric denominators

Both files are ignored by Git as part of `results/`. The report content
does not contain raw prompts or model completions.

## Reporting an existing experiment

First find its exact run ID under your local results folder. For example,
for a multi-scenario Lab07 run:

```powershell
uv run python -m lab_core.benchmark_report --lab lab07 --run-id YOUR_RUN_ID --scenario transient_failures --output results/benchmarks/lab07-report.md
```

To export JSON instead:

```powershell
uv run python -m lab_core.benchmark_report --lab lab07 --run-id YOUR_RUN_ID --scenario transient_failures --format json --output results/benchmarks/lab07-report.json
```

For an existing real-model run, explicitly select its lab and run ID:

```powershell
uv run python -m lab_core.benchmark_report --lab lab05 --run-id YOUR_RUN_ID --output results/benchmarks/lab05-report.md
```

The reporter scans local **saved data only**, so this command never calls
Groq. If you select a nonexistent run or omit `--scenario` when a Lab07
run contains multiple scenarios, it exits with an explanatory error.

## Reproduction rules

- Compare **within the same lab/run/scenario/measurement source and workload
  condition**, not across different prompts or experiments.
- Means are calculated from *observed* requests. If 20 calls were made but
  only 10 report cost, the cost denominator is `n=10`, not `n=20`.
- P50/P95 use nearest-rank on the measured latencies; they are descriptive,
  not statistical confidence intervals.
- Lab07 virtual time, simulated HTTP codes and client success are not
  evidence about real LLM API cost, token usage or answer quality.
- Identical seed and config reproduce the simulation behavior, but run ID,
  manifest timestamps, file paths and generated reports' run labels differ.
- Result files can contain sensitive information outside the allowlisted
  metrics. Keep `results/` private and git-ignored.
