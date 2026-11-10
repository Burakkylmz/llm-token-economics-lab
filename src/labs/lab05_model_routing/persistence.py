"""Per-run result isolation for Lab05 -- the same design as
labs.lab02-04's persistence modules, pointed at its own
`results/lab05_model_routing/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab05_model_routing.json`/`.csv`
files.

Like Lab02-04, Lab05 does NOT implement checkpoint/resume: a
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

from lab_core.config import Lab05Config
from labs.lab05_model_routing.models import RoutingMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
ROUTING_JSON_FILENAME = "routing_measurements.json"
ROUTING_CSV_FILENAME = "routing_measurements.csv"
TASK_COMPARISON_FILENAME = "task_comparison.csv"
STRATEGY_SUMMARY_FILENAME = "strategy_summary.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab05 does not support resume (see module
    docstring)."""


def lab05_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab05_model_routing" / experiment_run_id


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
    path = run_dir / ROUTING_JSON_FILENAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            if json.load(f):
                raise RunIdAlreadyUsed(
                    f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                    f"in '{ROUTING_JSON_FILENAME}'. Lab05 does not support resuming a run -- use a "
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


def build_initial_manifest(
    experiment_run_id: str,
    config: Lab05Config,
    profile: str,
    planned_calls: int,
    pricing_by_model: Optional[dict] = None,
) -> dict:
    # Snapshot the prices used at execution time so later re-pricing changes
    # do not silently alter the interpretation of this historical run.
    pricing = pricing_by_model or {}
    pricing_snapshot = {
        model: asdict(pricing[model]) if model in pricing else None
        for model in dict.fromkeys((config.small_model, config.large_model))
    }
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "pricing_snapshot": pricing_snapshot,
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
        writer.writerow(["execution_index", "strategy", "task_id"])
        for call in plan:
            writer.writerow([call.execution_index, call.strategy, call.task_id])


def write_routing_measurements(run_dir: Path, measurements: list[RoutingMeasurement]) -> None:
    json_path = run_dir / ROUTING_JSON_FILENAME
    csv_path = run_dir / ROUTING_CSV_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_routing_measurements(run_dir: Path) -> list[RoutingMeasurement]:
    path = run_dir / ROUTING_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(RoutingMeasurement)}
    return [RoutingMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_task_comparison_csv(run_dir: Path, comparisons) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / TASK_COMPARISON_FILENAME
    if not comparisons:
        return
    rows = [asdict(c) for c in comparisons]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_strategy_summary_csv(run_dir: Path, summaries) -> None:
    """`routing_rule_distribution`'s flattened keys differ per strategy
    (e.g. the two fixed baselines never match any rule, so their flattened
    key set is empty while the router's is not) -- fieldnames are the
    UNION across every row, not just the first, so a row with extra keys
    never raises DictWriter's "fields not in fieldnames" error."""
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / STRATEGY_SUMMARY_FILENAME
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
