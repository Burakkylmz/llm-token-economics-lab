"""Per-run result isolation for Lab08 -- the same design as every other
redesigned lab's persistence module, pointed at its own
`results/lab08_concurrency/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab08_concurrency.json`/`.csv`
files.

Like Lab02-07, Lab08 does NOT implement checkpoint/resume.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence

from lab_core.config import Lab08Config
from labs.lab08_concurrency.models import ConcurrencyMeasurement, RequestTimelineEvent
from labs.lab08_concurrency.stats import ConcurrencyComparison, ConditionSummary

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
RAW_JSON_FILENAME = "raw_measurements.json"
RAW_CSV_FILENAME = "raw_measurements.csv"
REQUEST_TIMELINE_FILENAME = "request_timeline.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
CONCURRENCY_COMPARISON_FILENAME = "concurrency_comparison.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab08 does not support resume."""


def lab08_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab08_concurrency" / experiment_run_id


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
    path = run_dir / RAW_JSON_FILENAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            if json.load(f):
                raise RunIdAlreadyUsed(
                    f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                    f"in '{RAW_JSON_FILENAME}'. Lab08 does not support resuming a run -- use a "
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


def build_initial_manifest(experiment_run_id: str, config: Lab08Config, profile: str, planned_calls: int) -> dict:
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
        writer.writerow(["execution_index", "repetition_index", "max_concurrent", "task_id"])
        for call in plan:
            writer.writerow([call.execution_index, call.repetition_index, call.max_concurrent, call.task_id])


def _write_dataclass_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_raw_measurements(run_dir: Path, measurements: Sequence[ConcurrencyMeasurement]) -> None:
    json_path = run_dir / RAW_JSON_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    _write_dataclass_csv(run_dir / RAW_CSV_FILENAME, [asdict(m) for m in measurements])


def read_raw_measurements(run_dir: Path) -> list[ConcurrencyMeasurement]:
    path = run_dir / RAW_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(ConcurrencyMeasurement)}
    return [ConcurrencyMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_request_timeline_csv(run_dir: Path, events: Sequence[RequestTimelineEvent]) -> None:
    _write_dataclass_csv(run_dir / REQUEST_TIMELINE_FILENAME, [asdict(e) for e in events])


def write_condition_summary_csv(run_dir: Path, summaries: Sequence[ConditionSummary]) -> None:
    rows = []
    fieldnames: list[str] = []
    for s in summaries:
        flat: dict = {}
        for key, value in asdict(s).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        rows.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    if not rows:
        return
    path = run_dir / CONDITION_SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_concurrency_comparison_csv(run_dir: Path, comparisons: Sequence[ConcurrencyComparison]) -> None:
    _write_dataclass_csv(run_dir / CONCURRENCY_COMPARISON_FILENAME, [asdict(c) for c in comparisons])


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
