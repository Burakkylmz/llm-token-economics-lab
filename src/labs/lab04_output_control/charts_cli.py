"""CLI wiring for `lab-cli lab04 backfill`: offline chart (re)generation.

Covers two independent, read-only sources (same pattern as
labs.lab03_prompt_caching.charts_cli):
  - per-run `results/lab04_output_control/<run_id>/` directories, via
    labs.lab04_output_control.charts.
  - the legacy flat `results/lab04_output_control.json` file, via
    labs.lab04_output_control.legacy_charts.

Never calls the Groq API, never needs GROQ_API_KEY, and never modifies any
existing JSON/CSV/Markdown result file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from labs.lab04_output_control.charts import generate_output_control_charts
from labs.lab04_output_control.legacy_charts import generate_legacy_output_control_charts

LAB04_SUBDIR = "lab04_output_control"
LEGACY_DIRNAME = "_legacy_backfill"


def add_backfill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        help="backfill mode only: regenerate charts for only this one per-run "
        "experiment_run_id (skips the legacy flat-file backfill). Default: every "
        "per-run directory found, plus the legacy flat-file backfill.",
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
    base = results_dir / LAB04_SUBDIR
    if not base.is_dir():
        return []
    if only_run_id is not None:
        candidate = base / only_run_id
        return [candidate] if candidate.is_dir() else []
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name != LEGACY_DIRNAME)


def run_backfill(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    print(f"LAB04 -- Chart Backfill (results dir: {results_dir})\n")

    total_written = 0
    for run_dir in _run_dirs(results_dir, args.run_id):
        written = generate_output_control_charts(run_dir)
        if written:
            print(f"[run]    {run_dir.name}: {len(written)} chart(s) -> {run_dir / 'charts'}")
            total_written += len(written)
        else:
            print(f"[run]    {run_dir.name}: no usable measurements, skipped")

    if args.run_id is None:
        legacy_written = generate_legacy_output_control_charts(results_dir)
        if legacy_written:
            legacy_dir = results_dir / LAB04_SUBDIR / LEGACY_DIRNAME / "charts"
            print(f"[legacy] {len(legacy_written)} chart(s) -> {legacy_dir}")
            total_written += len(legacy_written)
        else:
            print("[legacy] no usable legacy lab04_output_control.json measurements, skipped")

    print(f"\nTotal charts written: {total_written}")
    print("No API calls were made; no existing JSON/CSV/Markdown result files were modified.")
