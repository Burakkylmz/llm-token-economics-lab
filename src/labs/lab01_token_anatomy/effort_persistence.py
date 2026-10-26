"""Results isolation, manifest, and checkpoint/resume support for Lab01's
`effort` mode -- the same design as
labs.lab01_token_anatomy.factorial_persistence, pointed at its own
`results/lab01_effort/<run_id>/` directory so it never touches Lab01 basic
mode's outputs OR any pre-existing `results/lab01_factorial/` runs.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
from dataclasses import asdict, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from lab_core.config import Lab01EffortConfig
from labs.lab01_token_anatomy.effort_models import EffortMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
RAW_JSON_FILENAME = "raw_measurements.json"
RAW_CSV_FILENAME = "raw_measurements.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
EFFORT_SUMMARY_FILENAME = "effort_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class ResumeConfigurationMismatch(RuntimeError):
    """Raised when resuming a run whose on-disk config_fingerprint differs from
    the current configuration."""


def compute_config_fingerprint(config: Lab01EffortConfig, model: str) -> str:
    canonical = {
        "model": model,
        "prompts": sorted(config.prompts),
        "reasoning_efforts": sorted(config.reasoning_efforts),
        "repetitions": config.repetitions,
        "max_completion_tokens": config.max_completion_tokens,
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_git_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def effort_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab01_effort" / experiment_run_id


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def write_manifest(run_dir: Path, manifest: dict) -> None:
    _atomic_write_json(run_dir / MANIFEST_FILENAME, manifest)


def read_manifest(run_dir: Path) -> Optional[dict]:
    manifest_path = run_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        return None
    with open(manifest_path, encoding="utf-8") as f:
        return json.load(f)


def build_initial_manifest(
    experiment_run_id: str,
    config: Lab01EffortConfig,
    model: str,
    planned_conditions: int,
    planned_calls: int,
) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "model": model,
        "provider": "groq",
        "planned_conditions": planned_conditions,
        "planned_calls": planned_calls,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "stopped_reason": None,
        "configuration_fingerprint": compute_config_fingerprint(config, model),
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def read_existing_measurements(run_dir: Path) -> list[EffortMeasurement]:
    raw_path = run_dir / RAW_JSON_FILENAME
    if not raw_path.exists():
        return []
    with open(raw_path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(EffortMeasurement)}
    return [EffortMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def append_measurement(run_dir: Path, measurement: EffortMeasurement, is_first_write: bool) -> None:
    existing = read_existing_measurements(run_dir)
    existing.append(measurement)
    _atomic_write_json(run_dir / RAW_JSON_FILENAME, [asdict(m) for m in existing])

    csv_path = run_dir / RAW_CSV_FILENAME
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    row = asdict(measurement)
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def write_experiment_plan_csv(run_dir: Path, plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / PLAN_FILENAME
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["execution_index", "condition_id", "prompt_type", "reasoning_effort", "repetition_index", "api_seed"]
        )
        for call in plan:
            writer.writerow(
                [
                    call.execution_index,
                    call.condition.condition_id,
                    call.condition.prompt_type,
                    call.condition.reasoning_effort,
                    call.repetition_index,
                    call.api_seed,
                ]
            )


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


def write_effort_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EFFORT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def completed_pairs(measurements: list[EffortMeasurement]) -> set[tuple[str, int]]:
    return {(m.condition_id, m.repetition_index) for m in measurements if m.succeeded}


def validate_resume_fingerprint(run_dir: Path, config: Lab01EffortConfig, model: str) -> None:
    manifest = read_manifest(run_dir)
    if manifest is None:
        return
    current_fingerprint = compute_config_fingerprint(config, model)
    stored_fingerprint = manifest.get("configuration_fingerprint")
    if stored_fingerprint != current_fingerprint:
        raise ResumeConfigurationMismatch(
            f"Refusing to resume experiment_run_id '{manifest.get('experiment_run_id')}': "
            f"its stored configuration_fingerprint ({stored_fingerprint}) does not match "
            f"the current configuration's fingerprint ({current_fingerprint}). This usually "
            "means prompts/reasoning_efforts/repetitions/max_completion_tokens/model changed "
            "since that run started. Use a different --experiment-run-id to start a new, "
            "separate run."
        )
