"""Per-run result isolation for Lab07 -- the same design as every other
redesigned lab's persistence module, pointed at its own
`results/lab07_rate_limiting/<run_id>/` directory so it never touches or
overwrites the pre-existing flat `results/lab07_rate_limiting.json`/`.csv`
files.

Like Lab02-06, Lab07 does NOT implement checkpoint/resume.
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

from lab_core.config import Lab07Config
from labs.lab07_rate_limiting.models import AttemptRecord, ClientSimulationResult
from labs.lab07_rate_limiting.stats import StrategySummary

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
ATTEMPT_TIMELINE_FILENAME = "attempt_timeline.csv"
RAW_JSON_FILENAME = "raw_measurements.json"
RAW_CSV_FILENAME = "raw_measurements.csv"
STRATEGY_SUMMARY_FILENAME = "strategy_summary.csv"
SCENARIO_COMPARISON_FILENAME = "scenario_comparison.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab07 does not support resume."""


def lab07_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab07_rate_limiting" / experiment_run_id


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
                    f"in '{RAW_JSON_FILENAME}'. Lab07 does not support resuming a run -- use a "
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


def build_initial_manifest(experiment_run_id: str, config: Lab07Config, profile: str, planned_runs: int, seed: int) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "profile": profile,
        "randomization_seed": seed,
        "simulation_version": "v1.0-virtual-clock",
        "real_api_calls": 0,
        "real_token_usage": None,
        "real_api_cost": 0,
        "planned_simulation_runs": planned_runs,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, plan) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "scenario_id", "strategy_id", "client_count"])
        for call in plan:
            writer.writerow([call.execution_index, call.scenario_id, call.strategy_id, len(call.client_ids)])


def _write_dataclass_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_attempt_timeline_csv(run_dir: Path, records: Sequence[AttemptRecord]) -> None:
    _write_dataclass_csv(run_dir / ATTEMPT_TIMELINE_FILENAME, [asdict(r) for r in records])


def write_raw_measurements(run_dir: Path, results: Sequence[ClientSimulationResult]) -> None:
    json_path = run_dir / RAW_JSON_FILENAME
    _atomic_write_json(json_path, [asdict(r) for r in results])
    _write_dataclass_csv(run_dir / RAW_CSV_FILENAME, [asdict(r) for r in results])


def read_raw_measurements(run_dir: Path) -> list[ClientSimulationResult]:
    path = run_dir / RAW_JSON_FILENAME
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        records = json.load(f)
    field_names = {f.name for f in fields(ClientSimulationResult)}
    return [ClientSimulationResult(**{k: v for k, v in record.items() if k in field_names}) for record in records]


def write_strategy_summary_csv(run_dir: Path, summaries: Sequence[StrategySummary]) -> None:
    _write_dataclass_csv(run_dir / STRATEGY_SUMMARY_FILENAME, [asdict(s) for s in summaries])


def write_scenario_comparison_csv(run_dir: Path, summaries: Sequence[StrategySummary]) -> None:
    """A pivoted, strategy-per-row view of the same StrategySummary data,
    one column group per scenario -- convenient for the "which strategy
    wins in which scenario" comparison this lab's report/CLI makes."""
    by_strategy: dict[str, dict] = {}
    scenario_ids: list[str] = []
    for s in summaries:
        if s.scenario_id not in scenario_ids:
            scenario_ids.append(s.scenario_id)
        row = by_strategy.setdefault(s.strategy_id, {"strategy_id": s.strategy_id})
        row[f"{s.scenario_id}.success_rate"] = s.success_rate
        row[f"{s.scenario_id}.total_attempts"] = s.total_attempts
        row[f"{s.scenario_id}.total_429_responses"] = s.total_429_responses
        row[f"{s.scenario_id}.completion_time_seconds"] = s.completion_time_seconds

    if not by_strategy:
        return
    fieldnames = ["strategy_id"]
    for scenario_id in scenario_ids:
        fieldnames += [
            f"{scenario_id}.success_rate", f"{scenario_id}.total_attempts",
            f"{scenario_id}.total_429_responses", f"{scenario_id}.completion_time_seconds",
        ]
    path = run_dir / SCENARIO_COMPARISON_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(by_strategy.values())


def write_experiment_summary_json(run_dir: Path, summary_payload: dict) -> None:
    _atomic_write_json(run_dir / EXPERIMENT_SUMMARY_FILENAME, summary_payload)


def write_analysis_report(run_dir: Path, markdown_text: str) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / ANALYSIS_REPORT_FILENAME, "w", encoding="utf-8") as f:
        f.write(markdown_text)
