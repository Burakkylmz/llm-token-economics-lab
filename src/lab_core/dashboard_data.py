"""Read-only, conservative normalization of saved laboratory measurements.

This module intentionally imports no provider SDK or dashboard framework:
viewing results can never initiate an LLM request. Only explicitly
allowlisted measurement files are read. Free-form prompts, completions,
summaries, traces and credentials are never loaded or exported.
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

MAX_FILE_BYTES = 20_000_000
MAX_ROWS_PER_FILE = 50_000

# All modern labs are supported, including distinct sub-experiments.
# JSON takes precedence over CSV when both exist, to avoid double counting.
EXPERIMENT_SOURCES: dict[str, tuple[str, ...]] = {
    "lab01_effort": ("raw_measurements.json", "raw_measurements.csv"),
    "lab01_factorial": ("raw_measurements.json", "raw_measurements.csv"),
    "lab02_context_growth": (
        "context_measurements.json", "multi_turn_measurements.json",
    ),
    "lab03_prompt_caching": ("cache_measurements.json",),
    "lab04_output_control": (
        "output_limit_measurements.json", "invocation_measurements.json",
    ),
    "lab05_model_routing": ("routing_measurements.json",),
    "lab06_streaming": ("stream_measurements.json",),
    "lab07_rate_limiting": ("raw_measurements.json",),
    "lab08_concurrency": ("raw_measurements.json",),
    "lab09_batch_processing": (
        "sequential_measurements.json", "concurrent_measurements.json",
        "batch_measurements.json",
    ),
    "lab10_conversation_summarization": (
        "summary_measurements.csv", "followup_measurements.csv",
    ),
}

# Legacy one-file outputs use ExperimentResult's nested sections.
LEGACY_SOURCES = {f"lab{n:02d}": f"lab{n:02d}" for n in range(1, 11)}


def _numeric(value: Any, *, integral: bool = False) -> float | int | None:
    if value is None or isinstance(value, bool) or value == "":
        return None
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        return None
    if not math.isfinite(number) or number < 0:
        return None
    if integral:
        return int(number) if number.is_integer() else None
    return number


def _bool(value: Any) -> bool | None:
    if value is True or value == "True" or value == "true" or value == "1" or value == 1:
        return True
    if value is False or value == "False" or value == "false" or value == "0" or value == 0:
        return False
    return None


def _first(row: dict, keys: tuple[str, ...]) -> Any:
    """Return the first PRESENT nonblank field, preserving 0 and False."""
    for key in keys:
        value = row.get(key)
        if value is not None and value != "":
            return value
    return None


def _model(row: dict, manifest: dict) -> str:
    value = _first(row, ("selected_model", "model", "request_model", "metadata.model"))
    if value is None:
        config = manifest.get("experiment_configuration")
        if isinstance(config, dict):
            value = _first(config, ("model", "default_model"))
        if value is None:
            value = manifest.get("model")
    return str(value) if value is not None else "Unknown"


def _normalize(row: dict, *, lab: str, experiment: str, run_id: str,
               source: str, manifest: dict) -> dict:
    status = _first(row, ("request_status", "reconciliation_status", "stream_status", "outcome"))
    succeeded = _bool(row.get("succeeded"))
    if succeeded is None and status is not None:
        if str(status).upper() == "SUCCESS":
            succeeded = True
        elif str(status).upper() in {
            "RATE_LIMITED", "TIMEOUT", "CONNECTION_ERROR", "PROVIDER_ERROR",
            "CANCELLED", "UNKNOWN_ERROR", "FAILED", "ERROR",
        }:
            succeeded = False
    if status is None and succeeded is not None:
        status = "SUCCESS" if succeeded else "FAILED"

    cost_key = next(
        (key for key in (
            "estimated_cost_usd", "estimated_batch_cost_usd",
            "estimated_standard_cost_usd",
            "followup_estimated_cost_usd",
            "summary_generation_estimated_cost_usd",
            "tokens_and_cost.estimated_cost_usd",
        ) if row.get(key) is not None and row.get(key) != ""),
        None,
    )
    condition = _first(row, (
        "strategy_id", "strategy", "condition_id", "condition", "execution_mode", "call_purpose",
        "prompt_type", "metadata.experiment_id",
    ))
    return {
        "lab": lab,
        "experiment": experiment,
        "run_id": run_id,
        "source": source,
        "model": "Simulation (no LLM)" if lab == "lab07" else _model(row, manifest),
        "condition": str(condition) if condition is not None else "Unspecified",
        "scenario": str(row.get("scenario_id") or "Unspecified"),
        "timestamp": str(_first(row, ("timestamp", "metadata.timestamp")) or ""),
        "status": str(status) if status is not None else "Unknown",
        "simulated": lab == "lab07",
        "total_attempts": _numeric(row.get("total_attempts"), integral=True) if lab == "lab07" else None,
        "retry_count": _numeric(row.get("retry_count"), integral=True) if lab == "lab07" else None,
        "rate_limit_response_count": _numeric(row.get("rate_limit_response_count"), integral=True) if lab == "lab07" else None,
        "total_scheduled_wait_seconds": _numeric(row.get("total_scheduled_wait_seconds")) if lab == "lab07" else None,
        "virtual_completion_time_seconds": _numeric(row.get("virtual_completion_time_seconds")) if lab == "lab07" else None,
        "succeeded": succeeded,
        "answer_correct": _bool(row.get("answer_correct")),
        "prompt_tokens": _numeric(_first(row, (
            "prompt_tokens", "followup_prompt_tokens",
            "summary_generation_prompt_tokens", "tokens_and_cost.prompt_tokens",
        )), integral=True),
        "completion_tokens": _numeric(_first(row, (
            "completion_tokens", "followup_completion_tokens",
            "summary_generation_completion_tokens", "tokens_and_cost.completion_tokens",
        )), integral=True),
        "total_tokens": _numeric(_first(row, (
            "total_tokens", "followup_total_tokens",
            "summary_generation_total_tokens", "tokens_and_cost.total_tokens",
        )), integral=True),
        "estimated_cost_usd": _numeric(row.get(cost_key)) if cost_key else None,
        "cost_basis": (
            "Batch discounted estimate" if cost_key == "estimated_batch_cost_usd"
            else "Standard estimate" if cost_key == "estimated_standard_cost_usd"
            else "Estimated" if cost_key else "Unavailable"
        ),
        "latency_ms": _numeric(_first(row, (
            "total_latency_ms", "task_end_to_end_latency_ms",
            "total_stream_latency_ms", "non_streaming_total_latency_ms",
            "followup_latency_ms", "summary_generation_latency_ms",
            "performance.client_elapsed_ms",
        ))),
    }


def _read_rows(path: Path) -> list[dict]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("Not a regular measurement file")
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError("Measurement file exceeds the size limit")
    if path.suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            result = []
            for row in reader:
                if len(result) >= MAX_ROWS_PER_FILE:
                    raise ValueError("Measurement file exceeds the row limit")
                result.append(row)
            return result
    with path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, list) or len(payload) > MAX_ROWS_PER_FILE:
        raise ValueError("Measurement JSON must be a bounded list")
    if not all(isinstance(row, dict) for row in payload):
        raise ValueError("Measurement JSON contains non-record entries")
    return payload


def _manifest(run_dir: Path) -> dict:
    path = run_dir / "experiment_manifest.json"
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 100_000:
        return {}
    try:
        with path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        return payload if isinstance(payload, dict) else {}
    except (OSError, ValueError, UnicodeError):
        return {}


def _legacy_flatten(row: dict) -> dict:
    # Intentionally copy only the metric-bearing sections.
    flattened = {}
    for prefix in ("metadata", "tokens_and_cost", "performance"):
        section = row.get(prefix)
        if isinstance(section, dict):
            for key in ("model", "experiment_id", "timestamp", "prompt_tokens",
                        "completion_tokens", "total_tokens", "estimated_cost_usd",
                        "client_elapsed_ms"):
                if key in section:
                    flattened[f"{prefix}.{key}"] = section[key]
    return flattened


def scan_results(results_dir: Path) -> tuple[list[dict], list[str]]:
    """Return normalized measurements and nonfatal warnings, never modify files.

    The results root and known experiment directories are fixed; no path comes
    from row values, manifests or a browser-supplied filename.
    """
    records: list[dict] = []
    warnings: list[str] = []
    root = Path(results_dir)
    if not root.is_dir():
        return records, warnings

    for experiment, names in EXPERIMENT_SOURCES.items():
        base = root / experiment
        if base.is_symlink() or not base.is_dir():
            continue
        for run_dir in sorted(base.iterdir()):
            if run_dir.is_symlink() or not run_dir.is_dir():
                continue
            manifest = _manifest(run_dir)
            # Include one representation for each measurement family.
            seen: set[str] = set()
            for name in names:
                stem = Path(name).stem
                if stem in seen:
                    continue
                path = run_dir / name
                if not path.exists():
                    continue
                try:
                    rows = _read_rows(path)
                except (OSError, ValueError, UnicodeError, csv.Error) as exc:
                    warnings.append(f"{experiment}/{run_dir.name}/{name}: {exc}")
                    continue
                seen.add(stem)
                for row in rows:
                    records.append(_normalize(
                        row, lab=experiment.split("_", 1)[0], experiment=experiment,
                        run_id=run_dir.name, source=name, manifest=manifest,
                    ))

    # Support historical flat files without exposing their original text.
    for lab in LEGACY_SOURCES:
        for filename in (f"{lab}.json", f"{lab}_token_anatomy.json",
                         f"{lab}_rate_limiting.json"):
            path = root / filename
            if not path.exists():
                continue
            try:
                rows = _read_rows(path)
            except (OSError, ValueError, UnicodeError, csv.Error) as exc:
                warnings.append(f"{filename}: {exc}")
                continue
            for row in rows:
                flattened = _legacy_flatten(row)
                if flattened:
                    records.append(_normalize(
                        flattened, lab=lab, experiment="legacy", run_id="legacy",
                        source=filename, manifest={},
                    ))
            break
    return records, warnings


def filter_records(records: Iterable[dict], *, labs: Iterable[str] = (),
                   runs: Iterable[str] = (), models: Iterable[str] = (),
                   conditions: Iterable[str] = (), scenarios: Iterable[str] = ()) -> list[dict]:
    selected = [set(values) for values in (labs, runs, models, conditions, scenarios)]
    keys = ("lab", "run_id", "model", "condition", "scenario")
    return [
        row for row in records
        if all(not values or row[key] in values for key, values in zip(keys, selected))
    ]


def summarize(records: Iterable[dict], group_keys: tuple[str, ...] = (
    "lab", "model", "condition",
)) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for row in records:
        groups[tuple(row[key] for key in group_keys)].append(row)
    summary = []
    for key, group in sorted(groups.items()):
        known_costs = [r["estimated_cost_usd"] for r in group if r["estimated_cost_usd"] is not None]
        known_latencies = [r["latency_ms"] for r in group if r["latency_ms"] is not None]
        successes = [r["succeeded"] for r in group if r["succeeded"] is not None]
        correctness = [r["answer_correct"] for r in group if r["answer_correct"] is not None]
        token_values = [r["total_tokens"] for r in group if r["total_tokens"] is not None]
        summary.append({
            **dict(zip(group_keys, key)),
            "requests": len(group),
            "success_rate": sum(successes) / len(successes) if successes else None,
            "success_evaluated": len(successes),
            "accuracy": sum(correctness) / len(correctness) if correctness else None,
            "quality_evaluated": len(correctness),
            "estimated_cost_usd": sum(known_costs) if known_costs else None,
            "cost_observed": len(known_costs),
            "mean_latency_ms": sum(known_latencies) / len(known_latencies) if known_latencies else None,
            "latency_observed": len(known_latencies),
            "total_tokens": sum(token_values) if token_values else None,
            "token_observed": len(token_values),
        })
    return summary


def summarize_lab07(records: Iterable[dict]) -> list[dict]:
    """Group synthetic retry measurements by run, scenario and strategy.

    These are virtual scheduling measurements, not actual API latency,
    model accuracy or provider charges. Missing values are never imputed.
    """
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for record in records:
        if record.get("lab") == "lab07":
            groups[(record["run_id"], record["scenario"], record["condition"])].append(record)

    result = []
    for (run_id, scenario, strategy), group in sorted(groups.items()):
        def known(key):
            return [row[key] for row in group if row[key] is not None]

        success = known("succeeded")
        completions = known("virtual_completion_time_seconds")
        attempts = known("total_attempts")
        retries = known("retry_count")
        rate_limits = known("rate_limit_response_count")
        waits = known("total_scheduled_wait_seconds")
        result.append({
            "run_id": run_id,
            "scenario": scenario,
            "strategy": strategy,
            "simulated_clients": len(group),
            "success_rate_percent": 100 * sum(success) / len(success) if success else None,
            "successful_clients": sum(success) if success else None,
            "success_evaluated": len(success),
            "total_attempts": sum(attempts) if attempts else None,
            "total_retries": sum(retries) if retries else None,
            "total_429_responses": sum(rate_limits) if rate_limits else None,
            "total_virtual_wait_seconds": sum(waits) if waits else None,
            "mean_successful_virtual_completion_seconds": (
                sum(completions) / len(completions) if completions else None
            ),
        })
    return result


EXPORT_COLUMNS = (
    "lab", "experiment", "run_id", "source", "model", "condition", "scenario",
    "timestamp", "status", "succeeded", "answer_correct", "prompt_tokens",
    "completion_tokens", "total_tokens", "estimated_cost_usd", "cost_basis",
    "latency_ms", "simulated", "total_attempts", "retry_count",
    "rate_limit_response_count", "total_scheduled_wait_seconds",
    "virtual_completion_time_seconds",
)


def export_csv(records: Iterable[dict]) -> str:
    """Export only approved metrics, not prompts or provider response content."""
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=EXPORT_COLUMNS)
    writer.writeheader()
    for record in records:
        safe = {}
        for key in EXPORT_COLUMNS:
            value = record.get(key)
            if value is None:
                value = ""
            # Escape spreadsheet formulas in untrusted textual identifiers.
            if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                value = "'" + value
            safe[key] = value
        writer.writerow(safe)
    return stream.getvalue()
