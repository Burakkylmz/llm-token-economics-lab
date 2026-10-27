"""Per-run result isolation for Lab02 -- the same design as
labs.lab01_token_anatomy.effort_persistence, pointed at its own
`results/lab02_context_growth/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab02_context_growth.json`/`.csv`
files (a file and a same-stem directory are distinct filesystem entries;
this exact non-collision was already relied on and verified for Lab01's
basic-mode charts -- see labs.lab01_token_anatomy.basic_charts).

Unlike Lab01's effort mode, Lab02 does NOT implement checkpoint/resume (the
task's own scope-control guidance explicitly allows skipping this if it
would require a new framework): instead, `refuse_if_already_has_measurements`
simply refuses to execute into a run_id directory that already has
recorded measurements, so an accidental duplicate `--experiment-run-id` can
never silently append duplicate rows or overwrite completed data. Use a
different --experiment-run-id for a new run.
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

from lab_core.config import Lab02Config
from labs.lab02_context_growth.models import ContextMeasurement, TurnMeasurement

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
CONTEXT_JSON_FILENAME = "context_measurements.json"
CONTEXT_CSV_FILENAME = "context_measurements.csv"
MULTI_TURN_JSON_FILENAME = "multi_turn_measurements.json"
MULTI_TURN_CSV_FILENAME = "multi_turn_measurements.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab02 does not support resume (see module
    docstring), so this is a hard refusal rather than a silent append."""


def lab02_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab02_context_growth" / experiment_run_id


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


def _atomic_write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp_path, path)


def refuse_if_already_has_measurements(run_dir: Path) -> None:
    for filename in (CONTEXT_JSON_FILENAME, MULTI_TURN_JSON_FILENAME):
        path = run_dir / filename
        if path.exists():
            with open(path, encoding="utf-8") as f:
                if json.load(f):
                    raise RunIdAlreadyUsed(
                        f"experiment_run_id directory '{run_dir}' already has recorded measurements "
                        f"in '{filename}'. Lab02 does not support resuming a run -- use a different "
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
    config: Lab02Config,
    model: str,
    planned_context_calls: int,
    planned_turn_calls: int,
) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "model": model,
        "provider": "groq",
        "planned_context_calls": planned_context_calls,
        "planned_turn_calls": planned_turn_calls,
        "planned_calls": planned_context_calls + planned_turn_calls,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "stopped_reason": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, context_plan, turns: int) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["kind", "execution_index", "condition_id_or_turn", "repetition_index"])
        for call in context_plan:
            writer.writerow(["context", call.execution_index, call.condition_id, call.repetition_index])
        for turn in range(1, turns + 1):
            writer.writerow(["multi_turn_growth", None, turn, None])


def _write_measurements(json_path: Path, csv_path: Path, measurements: list) -> None:
    _atomic_write_json(json_path, [asdict(m) for m in measurements])
    if not measurements:
        return
    rows = [asdict(m) for m in measurements]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_context_measurements(run_dir: Path, measurements: list[ContextMeasurement]) -> None:
    _write_measurements(run_dir / CONTEXT_JSON_FILENAME, run_dir / CONTEXT_CSV_FILENAME, measurements)


def write_multi_turn_measurements(run_dir: Path, measurements: list[TurnMeasurement]) -> None:
    _write_measurements(run_dir / MULTI_TURN_JSON_FILENAME, run_dir / MULTI_TURN_CSV_FILENAME, measurements)


def read_context_measurements(run_dir: Path) -> list[ContextMeasurement]:
    path = run_dir / CONTEXT_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(ContextMeasurement)}
    return [ContextMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def read_multi_turn_measurements(run_dir: Path) -> list[TurnMeasurement]:
    path = run_dir / MULTI_TURN_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(TurnMeasurement)}
    return [TurnMeasurement(**{k: v for k, v in record.items() if k in field_names}) for record in records]


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
