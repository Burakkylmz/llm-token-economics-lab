"""Per-run result isolation for Lab03 -- the same design as
labs.lab02_context_growth.persistence, pointed at its own
`results/lab03_prompt_caching/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab03_prompt_caching.json`/`.csv`
files (a file and a same-stem directory are distinct filesystem entries).

Like Lab02, Lab03 does NOT implement checkpoint/resume: a `--experiment-
run-id` directory that already has recorded measurements is refused
outright rather than silently appended/overwritten -- use a different
--experiment-run-id to start a new, separate run.
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

from lab_core.config import Lab03Config
from labs.lab03_prompt_caching.models import CacheMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
CACHE_JSON_FILENAME = "cache_measurements.json"
CACHE_CSV_FILENAME = "cache_measurements.csv"
CACHE_DIAGNOSTICS_FILENAME = "cache_diagnostics.json"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab03 does not support resume (see module
    docstring)."""


def lab03_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab03_prompt_caching" / experiment_run_id


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
    path = run_dir / CACHE_JSON_FILENAME
    if path.exists():
        with open(path, encoding="utf-8") as f:
            if json.load(f):
                raise RunIdAlreadyUsed(
                    f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                    f"in '{CACHE_JSON_FILENAME}'. Lab03 does not support resuming a run -- use a "
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
    config: Lab03Config,
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
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "condition_id", "pair_index", "question_index"])
        for call in plan:
            writer.writerow([call.execution_index, call.condition_id, call.pair_index, call.question_index])


def write_cache_measurements(run_dir: Path, measurements: list[CacheMeasurement]) -> None:
    json_path = run_dir / CACHE_JSON_FILENAME
    csv_path = run_dir / CACHE_CSV_FILENAME
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_cache_diagnostics(run_dir: Path, measurements: list[CacheMeasurement]) -> None:
    """A bounded, safe projection of each measurement's diagnostic fields
    only (never prompt/completion text, never an API key). See
    labs.lab03_prompt_caching.diagnostics for what each field means and
    why; this function only assembles and writes what execute_planned_cache_call
    already computed onto each CacheMeasurement."""
    diagnostics = [
        {
            "provider_model": m.model,
            "sdk_version": m.sdk_version,
            "request_number": m.execution_index,
            "condition_id": m.condition_id,
            "succeeded": m.succeeded,
            "prompt_tokens": m.prompt_tokens,
            "prompt_tokens_details_present": m.prompt_tokens_details_present,
            "cached_tokens_field_present": m.cached_tokens_field_present,
            "cached_tokens": m.cached_tokens,
            "cache_measurement_status": m.cache_status,
            "raw_usage_field_names": m.raw_usage_field_names,
            "request_hash": m.request_hash,
        }
        for m in measurements
    ]
    _atomic_write_json(run_dir / CACHE_DIAGNOSTICS_FILENAME, diagnostics)


def read_cache_measurements(run_dir: Path) -> list[CacheMeasurement]:
    path = run_dir / CACHE_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(CacheMeasurement)}
    return [CacheMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


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


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
