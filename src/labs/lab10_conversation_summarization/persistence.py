"""Per-run result isolation for Lab10 -- same design as every other
redesigned lab's persistence module (Part 14.1), pointed at its own
`results/lab10_conversation_summarization/<run_id>/` directory.

Lab10 does NOT implement checkpoint/resume (same precedent as every other
lab in this project): reusing an `--experiment-run-id` that already has
recorded measurements is refused.
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

from lab_core.config import Lab10Config
from labs.lab10_conversation_summarization.models import FollowupMeasurement, PlannedCall, SummaryMeasurement, TraceRecord
from labs.lab10_conversation_summarization.stats import ConditionWorkflowCost, ContextComparisonRow

MANIFEST_FILENAME = "experiment_manifest.json"
PLAN_FILENAME = "experiment_plan.csv"
ORIGINAL_CONVERSATION_FILENAME = "original_conversation.json"
SUMMARIZATION_REQUESTS_FILENAME = "summarization_requests.json"
GENERATED_SUMMARIES_FILENAME = "generated_summaries.json"
INSPECTION_FILENAME = "request_response_inspection.json"
SUMMARY_MEASUREMENTS_FILENAME = "summary_measurements.csv"
FOLLOWUP_MEASUREMENTS_FILENAME = "followup_measurements.csv"
CONTEXT_COMPARISON_FILENAME = "context_comparison.csv"
TOKEN_COMPARISON_FILENAME = "token_comparison.csv"
COST_COMPARISON_FILENAME = "cost_comparison.csv"
QUALITY_EVALUATION_FILENAME = "quality_evaluation.json"
BREAK_EVEN_FILENAME = "break_even_analysis.csv"
EXPERIMENT_SUMMARY_FILENAME = "experiment_summary.json"
ANALYSIS_REPORT_FILENAME = "analysis_report.md"


class RunIdAlreadyUsed(RuntimeError):
    """Raised when --experiment-run-id points at a run_dir that already has
    recorded measurements -- Lab10 does not support resuming a run."""


def lab10_run_dir(results_dir: Path, experiment_run_id: str) -> Path:
    return results_dir / "lab10_conversation_summarization" / experiment_run_id


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
    path = run_dir / SUMMARY_MEASUREMENTS_FILENAME
    followup_path = run_dir / FOLLOWUP_MEASUREMENTS_FILENAME
    if path.exists() or followup_path.exists():
        raise RunIdAlreadyUsed(
            f"experiment_run_id directory '{run_dir}' already has recorded measurements. "
            "Lab10 does not support resuming a run -- use a different --experiment-run-id "
            "to start a new, separate run. Nothing was executed."
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
    experiment_run_id: str, config: Lab10Config, profile: str, planned_calls: int,
    dataset_version: str, dataset_hashes: dict, prompt_hashes: dict,
) -> dict:
    return {
        "experiment_run_id": experiment_run_id,
        "experiment_configuration": asdict(config),
        "profile": profile,
        "provider": "groq",
        "planned_calls": planned_calls,
        "dataset_version": dataset_version,
        "dataset_hashes": dataset_hashes,
        "prompt_hashes": prompt_hashes,
        "execution_started_at": datetime.now(timezone.utc).isoformat(),
        "execution_completed_at": None,
        "stopped_reason": None,
        "code_version_or_git_commit_if_available": get_git_commit(),
    }


def write_experiment_plan_csv(run_dir: Path, plan: Sequence[PlannedCall]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / PLAN_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["execution_index", "dataset_id", "condition_id", "call_purpose", "reasoning_effort", "question_id"])
        for call in plan:
            writer.writerow([call.execution_index, call.dataset_id, call.condition_id, call.call_purpose, call.reasoning_effort, call.question_id or ""])


def _write_dataclass_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_original_conversations(run_dir: Path, conversations: dict) -> None:
    payload = {
        dataset_id: {"scenario": c.scenario, "messages": [{"role": m.role, "content": m.content} for m in c.messages]}
        for dataset_id, c in conversations.items()
    }
    _atomic_write_json(run_dir / ORIGINAL_CONVERSATION_FILENAME, payload)


def write_generated_summaries(run_dir: Path, generated_summaries: dict[tuple[str, str], dict]) -> None:
    payload = {f"{dataset_id}::{condition_id}": summary for (dataset_id, condition_id), summary in generated_summaries.items()}
    _atomic_write_json(run_dir / GENERATED_SUMMARIES_FILENAME, payload)


def read_generated_summaries(run_dir: Path) -> dict[tuple[str, str], dict]:
    path = run_dir / GENERATED_SUMMARIES_FILENAME
    if not path.exists():
        return {}
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    result = {}
    for key, summary in payload.items():
        dataset_id, condition_id = key.split("::", 1)
        result[(dataset_id, condition_id)] = summary
    return result


def write_traces(run_dir: Path, traces: Sequence[TraceRecord]) -> None:
    all_traces = [asdict(t) for t in traces]
    _atomic_write_json(run_dir / INSPECTION_FILENAME, all_traces)
    summarization_only = [t for t in all_traces if t["call_purpose"] == "summary_generation"]
    _atomic_write_json(run_dir / SUMMARIZATION_REQUESTS_FILENAME, summarization_only)


def write_summary_measurements(run_dir: Path, measurements: Sequence[SummaryMeasurement]) -> None:
    rows = [asdict(m) for m in measurements]
    for row in rows:
        row["schema_errors"] = "|".join(row["schema_errors"])
    _write_dataclass_csv(run_dir / SUMMARY_MEASUREMENTS_FILENAME, rows)


def read_summary_measurements(run_dir: Path) -> list[SummaryMeasurement]:
    path = run_dir / SUMMARY_MEASUREMENTS_FILENAME
    if not path.exists():
        return []
    field_names = {f.name for f in fields(SummaryMeasurement)}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [SummaryMeasurement(**_coerce_row(row, field_names)) for row in rows]


def write_followup_measurements(run_dir: Path, measurements: Sequence[FollowupMeasurement]) -> None:
    _write_dataclass_csv(run_dir / FOLLOWUP_MEASUREMENTS_FILENAME, [asdict(m) for m in measurements])


def read_followup_measurements(run_dir: Path) -> list[FollowupMeasurement]:
    path = run_dir / FOLLOWUP_MEASUREMENTS_FILENAME
    if not path.exists():
        return []
    field_names = {f.name for f in fields(FollowupMeasurement)}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return [FollowupMeasurement(**_coerce_row(row, field_names)) for row in rows]


_INT_FIELDS = {
    "execution_index", "original_message_count", "original_conversation_character_count",
    "generated_summary_character_count", "summary_generation_prompt_tokens", "summary_generation_completion_tokens",
    "summary_generation_reasoning_tokens", "summary_generation_non_reasoning_completion_tokens",
    "summary_generation_cached_tokens", "summary_generation_total_tokens",
    "followup_prompt_tokens", "followup_completion_tokens", "followup_reasoning_tokens",
    "followup_non_reasoning_completion_tokens", "followup_cached_tokens", "followup_total_tokens",
}
_FLOAT_FIELDS = {
    "summary_generation_latency_ms", "summary_generation_estimated_cost_usd", "followup_latency_ms",
    "followup_estimated_cost_usd",
}
_BOOL_FIELDS = {"json_valid", "answer_correct"}
# json_valid has a real (non-Optional) `False` default on SummaryMeasurement;
# every other bool-typed field here (answer_correct) is Optional[bool] and
# must round-trip an empty CSV cell back to None, never a fabricated False.
_NON_OPTIONAL_BOOL_FIELDS = {"json_valid"}


def _coerce_row(row: dict, field_names: set[str]) -> dict:
    """CSV round-trips everything as strings -- restore None/int/float/bool
    so a backfilled run behaves identically to a freshly-executed one."""
    result = {}
    for key, value in row.items():
        if key not in field_names:
            continue
        if value == "" or value is None:
            result[key] = False if key in _NON_OPTIONAL_BOOL_FIELDS else None
            continue
        if key == "schema_errors":
            result[key] = tuple(v for v in value.split("|") if v) if value else ()
        elif key in _BOOL_FIELDS:
            result[key] = value in ("True", "true", "1")
        elif key in _INT_FIELDS:
            result[key] = int(value)
        elif key in _FLOAT_FIELDS:
            result[key] = float(value)
        else:
            result[key] = value
    return result


def write_context_comparison_csv(run_dir: Path, rows: Sequence[ContextComparisonRow]) -> None:
    _write_dataclass_csv(run_dir / CONTEXT_COMPARISON_FILENAME, [asdict(r) for r in rows])


def write_token_comparison_csv(run_dir: Path, rows: list[dict]) -> None:
    _write_dataclass_csv(run_dir / TOKEN_COMPARISON_FILENAME, rows)


def write_cost_comparison_csv(run_dir: Path, rows: Sequence[ConditionWorkflowCost], extra_fields: Optional[list[dict]] = None) -> None:
    base_rows = [asdict(r) for r in rows]
    if extra_fields:
        for row, extra in zip(base_rows, extra_fields):
            row.update(extra)
    _write_dataclass_csv(run_dir / COST_COMPARISON_FILENAME, base_rows)


def write_quality_evaluation(run_dir: Path, payload: dict) -> None:
    _atomic_write_json(run_dir / QUALITY_EVALUATION_FILENAME, payload)


def write_break_even_csv(run_dir: Path, rows: list[dict]) -> None:
    """`rows` is the full, already-flattened set of break-even rows across
    every dataset/condition -- always written as a single, complete
    replacement (never appended), since both `lab10 all` and `lab10
    backfill` always recompute the whole set together via analysis.py."""
    _write_dataclass_csv(run_dir / BREAK_EVEN_FILENAME, rows)


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
