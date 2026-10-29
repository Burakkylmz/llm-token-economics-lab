"""Per-run result isolation for Lab09, pointed at its own
`results/lab09_batch_processing/<run_id>/` directory -- same convention as
Lab08 (see labs.lab08_concurrency.persistence, the precedent this is
adapted from). Never touches or overwrites the pre-existing flat
`results/lab09_batch_processing.json`/`.csv` files from the original
implementation.

Lab09 does NOT implement checkpoint/resume for the real-time conditions
(same as Lab08), but its Batch job IS inherently resumable across process
restarts BY DESIGN: `batch_job.json` is the durable local state that
`submit_batch`/`check_batch_status`/`fetch_batch_results` each read back
in, so a participant can run those as three separate commands, possibly
days apart, without resubmitting anything.
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

from lab_core.config import Lab09Config
from labs.lab09_batch_processing.datasets import GoldenTask
from labs.lab09_batch_processing.models import (
    BatchJobMetadata,
    BatchRequestMeasurement,
    RealtimeMeasurement,
    RequestTimelineEvent,
)

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
WORKLOAD_SNAPSHOT_FILENAME = "golden_workload_snapshot.json"
BATCH_INPUT_FILENAME = "batch_input.jsonl"
BATCH_JOB_FILENAME = "batch_job.json"
SEQUENTIAL_JSON_FILENAME = "sequential_measurements.json"
SEQUENTIAL_CSV_FILENAME = "sequential_measurements.csv"
CONCURRENT_JSON_FILENAME = "concurrent_measurements.json"
CONCURRENT_CSV_FILENAME = "concurrent_measurements.csv"
REQUEST_TIMELINE_FILENAME = "request_timeline.csv"
BATCH_OUTPUT_FILENAME = "batch_output.jsonl"
BATCH_ERRORS_FILENAME = "batch_errors.jsonl"
BATCH_JSON_FILENAME = "batch_measurements.json"
BATCH_CSV_FILENAME = "batch_measurements.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
QUALITY_COMPARISON_FILENAME = "quality_comparison.csv"
COST_COMPARISON_FILENAME = "cost_comparison.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded real-time measurements -- the real-time conditions do not
    support resume (the Batch condition, by contrast, is always resumable
    via batch_job.json -- see module docstring)."""


def lab09_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab09_batch_processing" / experiment_run_id


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
    for filename in (SEQUENTIAL_JSON_FILENAME, CONCURRENT_JSON_FILENAME):
        path = run_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if json.load(f):
                    raise RunIdAlreadyUsed(
                        f"experiment_run_id directory '{run_dir}' already has recorded real-time "
                        f"measurements in '{filename}'. Use a different --experiment-run-id to start "
                        "a new, separate run. Nothing was executed."
                    )


def write_manifest(run_dir: Path, manifest: dict) -> None:
    _atomic_write_json(run_dir / MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> Optional[dict]:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_initial_manifest(experiment_run_id: str, config: Lab09Config, profile: str, planned_realtime_calls: int, batch_input_requests: int) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "profile": profile,
        "provider": "groq",
        "planned_realtime_calls": planned_realtime_calls,
        "batch_input_requests": batch_input_requests,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "condition", "max_concurrent", "task_id"])
        for call in plan:
            writer.writerow([call.execution_index, call.condition, call.max_concurrent, call.task_id])


def write_workload_snapshot(run_dir: Path, tasks: Sequence[GoldenTask]) -> None:
    _atomic_write_json(run_dir / WORKLOAD_SNAPSHOT_FILENAME, [asdict(t) for t in tasks])


def _write_dataclass_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_realtime_measurements(run_dir: Path, condition: str, measurements: Sequence[RealtimeMeasurement]) -> None:
    json_name, csv_name = (
        (SEQUENTIAL_JSON_FILENAME, SEQUENTIAL_CSV_FILENAME) if condition == "sequential"
        else (CONCURRENT_JSON_FILENAME, CONCURRENT_CSV_FILENAME)
    )
    rows = [asdict(m) for m in measurements]
    _atomic_write_json(run_dir / json_name, rows)
    _write_dataclass_csv(run_dir / csv_name, rows)


def read_realtime_measurements(run_dir: Path, condition: str) -> list[RealtimeMeasurement]:
    json_name = SEQUENTIAL_JSON_FILENAME if condition == "sequential" else CONCURRENT_JSON_FILENAME
    path = run_dir / json_name
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(RealtimeMeasurement)}
    return [RealtimeMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_request_timeline_csv(run_dir: Path, events: Sequence[RequestTimelineEvent]) -> None:
    _write_dataclass_csv(run_dir / REQUEST_TIMELINE_FILENAME, [asdict(e) for e in events])


def write_batch_job(run_dir: Path, job: BatchJobMetadata) -> None:
    """Atomic write -- the durable state `submit_batch`/`check_batch_status`/
    `fetch_batch_results` each read back, possibly in separate process
    invocations run days apart."""
    _atomic_write_json(run_dir / BATCH_JOB_FILENAME, asdict(job))


def read_batch_job(run_dir: Path) -> Optional[BatchJobMetadata]:
    path = run_dir / BATCH_JOB_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        record = json.load(f)
    field_names = {f.name for f in fields(BatchJobMetadata)}
    return BatchJobMetadata(**{k: v for k, v in record.items() if k in field_names})


def write_batch_raw_files(run_dir: Path, output_rows: list[dict], error_rows: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / BATCH_OUTPUT_FILENAME, "w", encoding="utf-8") as f:
        for row in output_rows:
            f.write(json.dumps(row) + "\n")
    with open(run_dir / BATCH_ERRORS_FILENAME, "w", encoding="utf-8") as f:
        for row in error_rows:
            f.write(json.dumps(row) + "\n")


def write_batch_measurements(run_dir: Path, measurements: Sequence[BatchRequestMeasurement]) -> None:
    rows = [asdict(m) for m in measurements]
    _atomic_write_json(run_dir / BATCH_JSON_FILENAME, rows)
    _write_dataclass_csv(run_dir / BATCH_CSV_FILENAME, rows)


def read_batch_measurements(run_dir: Path) -> list[BatchRequestMeasurement]:
    path = run_dir / BATCH_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(BatchRequestMeasurement)}
    return [BatchRequestMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def _flatten_dataclass_rows(objects) -> list[dict]:
    rows = []
    fieldnames: list[str] = []
    for obj in objects:
        flat: dict = {}
        for key, value in asdict(obj).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        rows.append(flat)
        for key in flat:
            if key not in fieldnames:
                fieldnames.append(key)
    return rows


def write_condition_summary_csv(run_dir: Path, summaries) -> None:
    rows = _flatten_dataclass_rows(summaries)
    if not rows:
        return
    path = run_dir / CONDITION_SUMMARY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_quality_comparison_csv(run_dir: Path, quality_summaries) -> None:
    _write_dataclass_csv(run_dir / QUALITY_COMPARISON_FILENAME, [asdict(q) for q in quality_summaries])


def write_cost_comparison_csv(run_dir: Path, pricing_comparison) -> None:
    _write_dataclass_csv(run_dir / COST_COMPARISON_FILENAME, [asdict(pricing_comparison)])


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def read_experiment_summary(run_dir: Path) -> Optional[dict]:
    path = run_dir / EXPERIMENT_SUMMARY_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
