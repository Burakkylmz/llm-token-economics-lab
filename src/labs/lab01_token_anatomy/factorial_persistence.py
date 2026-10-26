"""Results isolation, manifest, and checkpoint/resume support for Lab01's
factorial experiment engine.

Design choices, and why:
  - Every factorial run gets its own `results/lab01_factorial/<run_id>/`
    directory. Lab01's basic-mode outputs (lab01_token_anatomy.{json,csv}
    and its _summary.json) live one level up and are never touched by this
    module -- results isolation is just "a different directory", nothing
    more exotic.
  - No external database. Three small files per run:
      experiment_manifest.json  -- identity, config fingerprint, timing
      raw_measurements.json     -- every completed call, rewritten via an
                                    atomic temp-file + os.replace after each
                                    call (safe even if the process dies
                                    mid-write; cheap at this project's scale
                                    of <= a few thousand records)
      raw_measurements.csv      -- the same data, true append-only (one
                                    `open(..., "a")` + one row per call)
  - Resume = re-derive the full deterministic execution plan from the
    config (build_execution_plan is a pure function of it), then skip any
    (condition_id, repetition_index) that already has a *successful*
    measurement on disk. No separate "progress" file to go stale.
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

from lab_core.config import Lab01FactorialConfig
from labs.lab01_token_anatomy.factorial_models import FactorialMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
RAW_JSON_FILENAME = "raw_measurements.json"
RAW_CSV_FILENAME = "raw_measurements.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
FACTORIAL_SUMMARY_FILENAME = "factorial_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class ResumeConfigurationMismatch(RuntimeError):
    """Raised when resuming a run whose on-disk config_fingerprint differs from
    the current configuration -- refusing to silently mix two different
    experiment designs under one experiment_run_id."""


def compute_config_fingerprint(config: Lab01FactorialConfig, model: str) -> str:
    """A stable hash of everything that defines WHAT was planned (not how it
    was ordered -- re-randomizing order on resume is fine as long as the
    same set of conditions/repetitions is being filled in)."""
    canonical = {
        "model": model,
        "prompts": sorted(config.prompts),
        "reasoning_efforts": sorted(config.reasoning_efforts),
        "temperatures": sorted(config.temperatures),
        "top_p_values": sorted(config.top_p_values),
        "repetitions": config.repetitions,
        "max_completion_tokens": config.max_completion_tokens,
    }
    canonical_json = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def get_git_commit() -> Optional[str]:
    """Best-effort `git rev-parse HEAD`; None (never fabricated) if git or a
    repo isn't available -- this is diagnostic metadata, not a requirement."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def factorial_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab01_factorial" / experiment_run_id


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
    config: Lab01FactorialConfig,
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


def read_existing_measurements(run_dir: Path) -> list[FactorialMeasurement]:
    raw_path = run_dir / RAW_JSON_FILENAME
    if not raw_path.exists():
        return []
    with open(raw_path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(FactorialMeasurement)}
    return [FactorialMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def append_measurement(run_dir: Path, measurement: FactorialMeasurement, is_first_write: bool) -> None:
    """Persist one more completed call: atomically rewrite the JSON array,
    and append one row to the CSV (writing the header only on the very first
    row of a fresh run directory)."""
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
            ["execution_index", "condition_id", "prompt_type", "reasoning_effort", "temperature", "top_p", "repetition_index", "api_seed"]
        )
        for call in plan:
            writer.writerow(
                [
                    call.execution_index,
                    call.condition.condition_id,
                    call.condition.prompt_type,
                    call.condition.reasoning_effort,
                    call.condition.temperature,
                    call.condition.top_p,
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


def write_factorial_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / FACTORIAL_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)


def completed_pairs(measurements: list[FactorialMeasurement]) -> set[tuple[str, int]]:
    """(condition_id, repetition_index) pairs that already have a successful
    measurement -- these are skipped on resume, never re-run (and never
    duplicated)."""
    return {(m.condition_id, m.repetition_index) for m in measurements if m.succeeded}


def validate_resume_fingerprint(run_dir: Path, config: Lab01FactorialConfig, model: str) -> None:
    manifest = read_manifest(run_dir)
    if manifest is None:
        return  # Nothing to resume against yet -- this is effectively a fresh run.
    current_fingerprint = compute_config_fingerprint(config, model)
    stored_fingerprint = manifest.get("configuration_fingerprint")
    if stored_fingerprint != current_fingerprint:
        raise ResumeConfigurationMismatch(
            f"Refusing to resume experiment_run_id '{manifest.get('experiment_run_id')}': "
            f"its stored configuration_fingerprint ({stored_fingerprint}) does not match "
            f"the current configuration's fingerprint ({current_fingerprint}). This usually "
            "means prompts/reasoning_efforts/temperatures/top_p_values/repetitions/"
            "max_completion_tokens/model changed since that run started. Use a different "
            "--experiment-run-id to start a new, separate run."
        )
