"""Per-run result isolation for Lab04 -- the same design as
labs.lab02_context_growth.persistence/labs.lab03_prompt_caching.persistence,
pointed at its own `results/lab04_output_control/<run_id>/` directory so it
never touches or overwrites the pre-existing flat
`results/lab04_output_control.json`/`.csv` files.

Like Lab02/Lab03, Lab04 does NOT implement checkpoint/resume: a
`--experiment-run-id` directory that already has recorded measurements is
refused outright rather than silently appended/overwritten.
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

from lab_core.config import Lab04Config
from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
OUTPUT_LIMIT_JSON_FILENAME = "output_limit_measurements.json"
OUTPUT_LIMIT_CSV_FILENAME = "output_limit_measurements.csv"
INVOCATION_JSON_FILENAME = "invocation_measurements.json"
INVOCATION_CSV_FILENAME = "invocation_measurements.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
INVOCATION_COMPARISON_FILENAME = "invocation_comparison.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab04 does not support resume (see module
    docstring)."""


def lab04_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab04_output_control" / experiment_run_id


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
    for filename in (OUTPUT_LIMIT_JSON_FILENAME, INVOCATION_JSON_FILENAME):
        path = run_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if json.load(f):
                    raise RunIdAlreadyUsed(
                        f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                        f"in '{filename}'. Lab04 does not support resuming a run -- use a different "
                        "--experiment-run-id to start a new, separate run. Nothing was executed."
                    )


def write_manifest(run_dir: Path, manifest: dict) -> None:
    _atomic_write_json(run_dir / MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> Optional[dict]:
    path = run_dir / MANIFEST_FILENAME
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_initial_manifest(
    experiment_run_id: str,
    config: Lab04Config,
    model: str,
    planned_calls: int,
) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "model": model,
        "provider": "groq",
        "planned_calls": planned_calls,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "stopped_reason": None,
        "experiment_mode": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, output_limit_plan, invocation_plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "phase", "condition_id_or_strategy", "question_index", "max_completion_tokens"])
        for call in output_limit_plan:
            writer.writerow([call.execution_index, "output_limit_sweep", call.condition_id, "", call.max_completion_tokens])
        for call in invocation_plan:
            writer.writerow([call.execution_index, "invocation_comparison", call.strategy, call.question_index, ""])


def write_output_limit_measurements(run_dir: Path, measurements: list[OutputLimitMeasurement]) -> None:
    json_path = run_dir / OUTPUT_LIMIT_JSON_FILENAME
    csv_path = run_dir / OUTPUT_LIMIT_CSV_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_invocation_measurements(run_dir: Path, measurements: list[InvocationMeasurement]) -> None:
    json_path = run_dir / INVOCATION_JSON_FILENAME
    csv_path = run_dir / INVOCATION_CSV_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_output_limit_measurements(run_dir: Path) -> list[OutputLimitMeasurement]:
    path = run_dir / OUTPUT_LIMIT_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(OutputLimitMeasurement)}
    return [OutputLimitMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_condition_summary_csv(run_dir: Path, summaries) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / CONDITION_SUMMARY_FILENAME
    if not summaries:
        return
    rows = []
    for summary in summaries:
        flat: dict = {}
        for key, value in asdict(summary).items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    flat[f"{key}.{sub_key}"] = sub_value
            else:
                flat[key] = value
        rows.append(flat)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_invocation_comparison_csv(run_dir: Path, comparison) -> None:
    """Flattens InvocationComparisonSummary (two nested StrategySummary
    objects) into one row per strategy plus the comparison fields, so the
    file is readable as a plain table."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / INVOCATION_COMPARISON_FILENAME
    if comparison is None:
        return
    rows = []
    for strategy_summary in (comparison.single, comparison.multiple):
        row = {k: v for k, v in asdict(strategy_summary).items() if k != "individual_request_latencies_ms"}
        row["individual_request_latencies_ms"] = ";".join(
            "" if v is None else str(v) for v in strategy_summary.individual_request_latencies_ms
        )
        row["absolute_cost_difference_usd"] = comparison.absolute_cost_difference_usd
        row["relative_cost_difference_percent"] = comparison.relative_cost_difference_percent
        row["workflow_latency_difference_ms"] = comparison.workflow_latency_difference_ms
        row["quality_comparison"] = comparison.quality_comparison
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
