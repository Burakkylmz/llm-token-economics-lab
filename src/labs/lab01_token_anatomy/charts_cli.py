"""CLI wiring for `lab-cli lab01 backfill`: offline chart (re)generation.

Scans existing `results/lab01_effort/<run_id>/` and
`results/lab01_factorial/<run_id>/` directories and renders PNG charts for
each into a `charts/` subdirectory, purely from already-written
`effort_summary.json` / `factorial_summary.json` files. Never calls the Groq
API, never needs GROQ_API_KEY, and never modifies any existing JSON/CSV/
Markdown result file -- this is a read-existing-results, write-new-PNGs-only
operation, safe to run as many times as needed (each run's charts are
simply overwritten with the same deterministic input).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from labs.lab01_token_anatomy.effort_charts import generate_effort_charts
from labs.lab01_token_anatomy.factorial_charts import generate_factorial_charts

EFFORT_SUBDIR = "lab01_effort"
FACTORIAL_SUBDIR = "lab01_factorial"


def add_backfill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        help="backfill mode only: regenerate charts for only this one run_id "
        "(searched in both lab01_effort and lab01_factorial). Default: every run found.",
    )
    parser.add_argument(
        "--effort-only",
        action="store_true",
        help="backfill mode only: skip legacy lab01_factorial results.",
    )
    parser.add_argument(
        "--factorial-only",
        action="store_true",
        help="backfill mode only: skip lab01_effort results.",
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


def _run_dirs(results_dir: Path, subdir: str, only_run_id: str | None) -> list[Path]:
    base = results_dir / subdir
    if not base.is_dir():
        return []
    if only_run_id is not None:
        candidate = base / only_run_id
        return [candidate] if candidate.is_dir() else []
    return sorted(p for p in base.iterdir() if p.is_dir())


def run_backfill(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    print(f"LAB01 -- Chart Backfill (results dir: {results_dir})\n")

    total_written = 0

    if not args.factorial_only:
        effort_dirs = _run_dirs(results_dir, EFFORT_SUBDIR, args.run_id)
        for run_dir in effort_dirs:
            written = generate_effort_charts(run_dir)
            if written:
                print(f"[effort]    {run_dir.name}: {len(written)} chart(s) -> {run_dir / 'charts'}")
                total_written += len(written)
            else:
                print(f"[effort]    {run_dir.name}: no usable measurements, skipped")

    if not args.effort_only:
        factorial_dirs = _run_dirs(results_dir, FACTORIAL_SUBDIR, args.run_id)
        for run_dir in factorial_dirs:
            written = generate_factorial_charts(run_dir)
            if written:
                print(f"[factorial] {run_dir.name}: {len(written)} chart(s) -> {run_dir / 'charts'}")
                total_written += len(written)
            else:
                print(f"[factorial] {run_dir.name}: no usable measurements, skipped")

    print(f"\nTotal charts written: {total_written}")
    print("No API calls were made; no existing JSON/CSV/Markdown result files were modified.")
