"""CLI wiring for `lab-cli lab03 backfill`: offline chart (re)generation.

Covers two independent, read-only sources:
  - per-run `results/lab03_prompt_caching/<run_id>/` directories, via
    labs.lab03_prompt_caching.charts.
  - the legacy flat `results/lab03_prompt_caching.json` file, via
    labs.lab03_prompt_caching.legacy_charts.

Never calls the Groq API, never needs GROQ_API_KEY, and never modifies
any existing JSON/CSV/Markdown result file.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from labs.lab03_prompt_caching.charts import generate_cache_charts
from labs.lab03_prompt_caching.legacy_charts import generate_legacy_cache_charts

LAB03_SUBDIR = "lab03_prompt_caching"
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
    base = results_dir / LAB03_SUBDIR
    if not base.is_dir():
        return []
    if only_run_id is not None:
        candidate = base / only_run_id
        return [candidate] if candidate.is_dir() else []
    return sorted(p for p in base.iterdir() if p.is_dir() and p.name != LEGACY_DIRNAME)


def run_backfill(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    print(f"LAB03 -- Chart Backfill (results dir: {results_dir})\n")

    total_written = 0
    for run_dir in _run_dirs(results_dir, args.run_id):
        written = generate_cache_charts(run_dir)
        if written:
            print(f"[run]    {run_dir.name}: {len(written)} chart(s) -> {run_dir / 'charts'}")
            total_written += len(written)
        else:
            print(f"[run]    {run_dir.name}: no usable measurements, skipped")

    if args.run_id is None:
        legacy_written = generate_legacy_cache_charts(results_dir)
        if legacy_written:
            legacy_dir = results_dir / LAB03_SUBDIR / LEGACY_DIRNAME / "charts"
            print(f"[legacy] {len(legacy_written)} chart(s) -> {legacy_dir}")
            total_written += len(legacy_written)
        else:
            print("[legacy] no usable legacy lab03_prompt_caching.json measurements, skipped")

    print(f"\nTotal charts written: {total_written}")
    print("No API calls were made; no existing JSON/CSV/Markdown result files were modified.")
