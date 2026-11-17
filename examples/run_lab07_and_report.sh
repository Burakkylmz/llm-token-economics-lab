#!/usr/bin/env bash
# Offline Lab07, no Groq API key needed.
# Run from repository root: bash examples/run_lab07_and_report.sh
set -euo pipefail
test -f pyproject.toml || { echo "Run from the repository root" >&2; exit 2; }
run_id="lab07-repro-$(date -u +%Y%m%dT%H%M%SZ)-$$"
uv run lab-cli lab07 all --profile pilot --seed 42 --experiment-run-id "$run_id"
mkdir -p results/benchmarks
uv run python -m lab_core.benchmark_report --lab lab07 --run-id "$run_id" --scenario transient_failures --format md --output "results/benchmarks/$run_id.md"
uv run python -m lab_core.benchmark_report --lab lab07 --run-id "$run_id" --scenario transient_failures --format json --output "results/benchmarks/$run_id.json"
printf 'Saved reports to results/benchmarks/%s.{md,json}\n' "$run_id"
