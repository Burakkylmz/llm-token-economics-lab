"""CLI wiring for `lab-cli lab10 backfill`: offline recomputation of every
derived report/CSV/JSON file plus charts, from already-persisted raw
measurements (Part 14.3). Never calls the Groq API, never needs
GROQ_API_KEY, never regenerates a summary or a follow-up answer.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from labs.lab10_conversation_summarization import persistence
from labs.lab10_conversation_summarization.analysis import compute_reports, persist_computed_reports
from labs.lab10_conversation_summarization.charts import generate_summarization_charts

LAB10_SUBDIR = "lab10_conversation_summarization"


def add_backfill_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--run-id",
        default=None,
        help="backfill mode only: recompute reports/charts for only this one "
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
    base = results_dir / LAB10_SUBDIR
    if not base.is_dir():
        return []
    if only_run_id is not None:
        candidate = base / only_run_id
        return [candidate] if candidate.is_dir() else []
    return sorted(p for p in base.iterdir() if p.is_dir())


def run_backfill(args: argparse.Namespace) -> None:
    results_dir = _resolve_results_dir(args)
    print(f"LAB10 -- Report & Chart Backfill (results dir: {results_dir})\n")

    total_charts = 0
    for run_dir in _run_dirs(results_dir, args.run_id):
        manifest = persistence.read_manifest(run_dir)
        if manifest is None:
            print(f"[run] {run_dir.name}: no manifest found, skipped")
            continue

        summary_measurements = persistence.read_summary_measurements(run_dir)
        followup_measurements = persistence.read_followup_measurements(run_dir)
        generated_summaries = persistence.read_generated_summaries(run_dir)
        if not summary_measurements and not followup_measurements:
            print(f"[run] {run_dir.name}: no recorded measurements, skipped")
            continue

        dataset_version = manifest.get("dataset_version", "v1")
        computed = compute_reports(manifest, summary_measurements, followup_measurements, generated_summaries, dataset_version)
        persist_computed_reports(run_dir, manifest, computed)

        chart_paths = generate_summarization_charts(run_dir)
        total_charts += len(chart_paths)
        print(f"[run] {run_dir.name}: reports recomputed, {len(chart_paths)} chart(s) -> {run_dir / 'charts'}")

    print(f"\nTotal charts written: {total_charts}")
    print("No API calls were made; no raw measurement file was modified or regenerated.")
