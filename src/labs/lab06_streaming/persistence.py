"""Per-run result isolation for Lab06 -- the same design as every other
redesigned lab's persistence module, pointed at its own
`results/lab06_streaming/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab06_streaming.json`/`.csv`
files.

Like Lab02-05, Lab06 does NOT implement checkpoint/resume.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab_core.config import Lab06Config
from labs.lab06_streaming.models import StreamEventRecord, StreamMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
STREAM_JSON_FILENAME = "stream_measurements.json"
STREAM_CSV_FILENAME = "stream_measurements.csv"
STREAM_EVENTS_FILENAME = "stream_events.csv"
PAIRED_COMPARISON_FILENAME = "paired_comparison.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab06 does not support resume."""


def lab06_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab06_streaming" / experiment_run_id


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def refuse_if_already_has_measurements(run_dir: Path) -> None:
    path = run_dir / STREAM_JSON_FILENAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            if json.load(f):
                raise RunIdAlreadyUsed(
                    f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                    f"in '{STREAM_JSON_FILENAME}'. Lab06 does not support resuming a run -- use a "
                    "different --experiment-run-id to start a new, separate run. Nothing was executed."
                )


def write_manifest(run_dir: Path, manifest: dict) -> None:
    _atomic_write_json(run_dir / MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> Optional[dict]:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_initial_manifest(experiment_run_id: str, config: Lab06Config, profile: str, planned_calls: int) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "profile": profile,
        "provider": "groq",
        "planned_calls": planned_calls,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "stopped_reason": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "pair_id", "condition"])
        for call in plan:
            writer.writerow([call.execution_index, call.pair_id, call.condition])


def write_stream_measurements(run_dir: Path, measurements: list[StreamMeasurement]) -> None:
    json_path = run_dir / STREAM_JSON_FILENAME
    csv_path = run_dir / STREAM_CSV_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_stream_measurements(run_dir: Path) -> list[StreamMeasurement]:
    path = run_dir / STREAM_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(StreamMeasurement)}
    return [StreamMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_stream_events_csv(run_dir: Path, events: list[StreamEventRecord]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / STREAM_EVENTS_FILENAME
    if not events:
        return
    rows = [asdict(e) for e in events]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_paired_comparison_csv(run_dir: Path, comparisons) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PAIRED_COMPARISON_FILENAME
    if not comparisons:
        return
    rows = [asdict(c) for c in comparisons]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_condition_summary_csv(run_dir: Path, summaries) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CONDITION_SUMMARY_FILENAME
    if not summaries:
        return
    rows = []
    fieldnames: list[str] = []
    for summary in summaries:
        flat: dict = {}
        for key, value in asdict(summary).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        rows.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
