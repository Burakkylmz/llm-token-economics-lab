"""CLI wiring for `lab-cli lab09 backfill`: offline chart (re)generation
from an existing run's `experiment_summary.json`. Never calls the Groq
API, never needs GROQ_API_KEY, and never modifies any existing JSON/CSV/
Markdown result file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from labs.lab09_batch_processing.charts import generate_batch_charts

LAB09_SUBDIR = "lab09_batch_processing"


def add_backfill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        help="backfill mode only: regenerate charts for only this one per-run "
        "experiment_run_id. Default: every per-run directory found.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="backfill mode only: override the results directory (default: "
        "experiment.results_dir from --config, or 'results').",
    )


def _resolve_results_dir(args: argparse.Namespace) -> Path:
    if args.results_dir is not None:
        return Path(args.results_dir)
    config = load_config(args.config)
    return Path(config.experiment.results_dir)


def _run_dirs(results_dir: Path, only_run_id: str | None) -> list[Path]:
    base = results_dir / LAB09_SUBDIR
    if not base.is_dir():
        return []
    if only_run_id is not None:
        candidate = base / only_run_id
        return [candidate] if candidate.is_dir() else []
    return sorted(p for p in base.iterdir() if p.is_dir())


def run_backfill(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    print(f"LAB09 -- Chart Backfill (results dir: {results_dir})\n")

    total_written = 0
    for run_dir in _run_dirs(results_dir, args.run_id):
        written = generate_batch_charts(run_dir)
        if written:
            print(f"[run]    {run_dir.name}: {len(written)} chart(s) -> {run_dir / 'charts'}")
            total_written += len(written)
        else:
            print(f"[run]    {run_dir.name}: no usable measurements, skipped")

    print(f"\nTotal charts written: {total_written}")
    print("No API calls were made; no existing JSON/CSV/Markdown result files were modified.")
