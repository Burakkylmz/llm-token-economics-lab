"""Opt-in, offline validation of saved experimental measurements.

The first explicit versioned contract applies to the *normalized* read-only
metric view. Existing heterogeneous raw formats remain supported as unversioned
inputs. Nothing in this module starts an experiment or calls a provider.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from lab_core.dashboard_data import (
    EXPERIMENT_SOURCES,
    EXPORT_COLUMNS,
    MAX_FILE_BYTES,
    _bool,
    _normalize,
    _numeric,
    _read_rows,
)

NORMALIZED_CONTRACT_VERSION = 1
VALIDATION_REPORT_VERSION = 1
VERSION_FIELD = "normalized_contract_version"
MAX_MANIFEST_BYTES = 100_000
MAX_ISSUES = 100

# The public normalized v1 field contract is independent of any particular
# laboratory's raw JSON/CSV shape. Exact names are guarded by unit tests.
TEXT_FIELDS = frozenset((
    "lab", "experiment", "run_id", "source", "model", "condition",
    "scenario", "timestamp", "status", "cost_basis",
))
BOOLEAN_FIELDS = frozenset(("simulated", "succeeded", "answer_correct"))
INTEGER_FIELDS = frozenset((
    "prompt_tokens", "completion_tokens", "total_tokens",
    "total_attempts", "retry_count", "rate_limit_response_count",
))
NUMBER_FIELDS = frozenset((
    "estimated_cost_usd", "latency_ms", "total_scheduled_wait_seconds",
    "virtual_completion_time_seconds",
))
OPTIONAL_FIELDS = frozenset(EXPORT_COLUMNS) - {
    "lab", "experiment", "run_id", "source", "model", "condition",
    "scenario", "timestamp", "status", "cost_basis", "simulated",
}

# Optional raw aliases. Only validate *present* values; a missing value is
# unknown, not zero. These are raw-input checks, before normalization can
# legitimately turn malformed values into None.
RAW_INTEGER_FIELDS = frozenset((
    "prompt_tokens", "completion_tokens", "total_tokens",
    "followup_prompt_tokens", "followup_completion_tokens", "followup_total_tokens",
    "summary_generation_prompt_tokens", "summary_generation_completion_tokens",
    "summary_generation_total_tokens", "tokens_and_cost.prompt_tokens",
    "tokens_and_cost.completion_tokens", "tokens_and_cost.total_tokens",
    "total_attempts", "retry_count", "rate_limit_response_count",
    "client_id", "attempt_number",
))
RAW_NUMBER_FIELDS = frozenset((
    "estimated_cost_usd", "estimated_batch_cost_usd", "estimated_standard_cost_usd",
    "followup_estimated_cost_usd", "summary_generation_estimated_cost_usd",
    "tokens_and_cost.estimated_cost_usd", "total_latency_ms",
    "task_end_to_end_latency_ms", "total_stream_latency_ms",
    "non_streaming_total_latency_ms", "followup_latency_ms",
    "summary_generation_latency_ms", "performance.client_elapsed_ms",
    "total_scheduled_wait_seconds", "virtual_completion_time_seconds",
))
RAW_BOOLEAN_FIELDS = frozenset(("succeeded", "answer_correct"))
LAB07_REQUIRED_V1 = (
    "scenario_id", "strategy_id", "client_id", "succeeded",
    "total_attempts", "retry_count", "rate_limit_response_count",
    "total_scheduled_wait_seconds",
)


@dataclass(frozen=True)
class Diagnostic:
    severity: str
    code: str
    source: str
    row: int | None = None
    field: str | None = None

    # No raw input values or prompt/response texts may reach diagnostics.


@dataclass
class ValidationResult:
    experiment: str
    run_id: str
    compatibility: str = "unknown"
    declared_normalized_contract_version: int | None = None
    file_count: int = 0
    record_count: int = 0
    diagnostics: list[Diagnostic] = field(default_factory=list)
    truncated_diagnostics: int = 0

    def issue(self, severity: str, code: str, source: str, *,
              row: int | None = None, field: str | None = None) -> None:
        if len(self.diagnostics) < MAX_ISSUES:
            self.diagnostics.append(Diagnostic(severity, code, source, row, field))
        else:
            self.truncated_diagnostics += 1

    @property
    def errors(self) -> int:
        return sum(item.severity == "error" for item in self.diagnostics)

    @property
    def warnings(self) -> int:
        return sum(item.severity == "warning" for item in self.diagnostics)

    def to_report(self) -> dict[str, Any]:
        return {
            "validation_report_version": VALIDATION_REPORT_VERSION,
            "normalized_contract_version": NORMALIZED_CONTRACT_VERSION,
            "experiment": self.experiment,
            "run_id": self.run_id,
            "declared_normalized_contract_version": self.declared_normalized_contract_version,
            "compatibility": self.compatibility,
            "valid": self.errors == 0 and self.truncated_diagnostics == 0,
            "files_checked": self.file_count,
            "records_checked": self.record_count,
            "errors": self.errors,
            "warnings": self.warnings,
            "diagnostics_truncated": self.truncated_diagnostics,
            "diagnostics": [asdict(item) for item in self.diagnostics],
        }


def _safe_run_id(value: str) -> bool:
    """Disallow traversal, separators, drive letters and hidden relative paths."""
    return bool(value) and value not in (".", "..") and not any(
        char in value for char in ("/", "\\", "\x00", ":")
    )


def _manifest_version(run_dir: Path, result: ValidationResult) -> dict:
    path = run_dir / "experiment_manifest.json"
    if not path.exists():
        result.compatibility = "legacy_unversioned"
        result.issue("warning", "unversioned_manifest", "experiment_manifest.json")
        return {}
    if path.is_symlink() or not path.is_file():
        result.compatibility = "invalid_manifest"
        result.issue("error", "unsafe_manifest", "experiment_manifest.json")
        return {}
    try:
        if path.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("Manifest too large")
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        if not isinstance(manifest, dict):
            raise ValueError("Manifest must be an object")
    except (OSError, ValueError, UnicodeError):
        result.compatibility = "invalid_manifest"
        result.issue("error", "invalid_manifest", "experiment_manifest.json")
        return {}

    declared = manifest.get(VERSION_FIELD)
    if declared is None:
        result.compatibility = "legacy_unversioned"
        result.issue("warning", "unversioned_manifest", "experiment_manifest.json")
    elif type(declared) is not int or declared < 1:
        result.compatibility = "unsupported"
        result.issue("error", "invalid_contract_version", "experiment_manifest.json", field=VERSION_FIELD)
    elif declared != NORMALIZED_CONTRACT_VERSION:
        result.declared_normalized_contract_version = declared
        result.compatibility = "unsupported"
        result.issue("error", "unsupported_contract_version", "experiment_manifest.json", field=VERSION_FIELD)
    else:
        result.declared_normalized_contract_version = declared
        result.compatibility = "v1"
    return manifest


def _validate_raw(row: dict, *, source: str, index: int, result: ValidationResult) -> None:
    if None in row:
        result.issue("error", "malformed_csv_columns", source, row=index)
    for name in sorted(RAW_INTEGER_FIELDS | RAW_NUMBER_FIELDS | RAW_BOOLEAN_FIELDS):
        if name not in row or row[name] is None or row[name] == "":
            continue
        if name in RAW_BOOLEAN_FIELDS:
            valid = _bool(row[name]) is not None
        else:
            valid = _numeric(row[name], integral=name in RAW_INTEGER_FIELDS) is not None
        if not valid:
            result.issue("error", "invalid_raw_value", source, row=index, field=name)
    if source == "raw_measurements.json" and result.experiment == "lab07_rate_limiting":
        # New v1 Lab07 runs must have their actual per-client identifiers.
        # Old unversioned fixtures remain readable (warning-level only).
        for name in LAB07_REQUIRED_V1:
            if name not in row or row[name] is None or row[name] == "":
                severity = "error" if result.compatibility == "v1" else "warning"
                result.issue(severity, "missing_lab07_field", source, row=index, field=name)


def _validate_normalized(row: dict, *, source: str, index: int,
                         result: ValidationResult) -> None:
    if set(row) != set(EXPORT_COLUMNS):
        result.issue("error", "normalized_field_mismatch", source, row=index)
        return
    for key, value in row.items():
        if value is None:
            if key not in OPTIONAL_FIELDS:
                result.issue("error", "missing_normalized_field", source, row=index, field=key)
            continue
        if key in TEXT_FIELDS:
            ok = isinstance(value, str)
        elif key in BOOLEAN_FIELDS:
            ok = type(value) is bool
        elif key in INTEGER_FIELDS:
            ok = type(value) is int and value >= 0
        elif key in NUMBER_FIELDS:
            ok = (type(value) in (float, int) and math.isfinite(value) and value >= 0)
        else:
            ok = False
        if not ok:
            result.issue("error", "normalized_type_mismatch", source, row=index, field=key)

    prompt, completion, total = (
        row["prompt_tokens"], row["completion_tokens"], row["total_tokens"]
    )
    if all(type(v) is int for v in (prompt, completion, total)):
        if prompt + completion != total:
            result.issue("warning", "token_arithmetic_mismatch", source, row=index, field="total_tokens")
    if result.experiment == "lab07_rate_limiting":
        if not row["simulated"]:
            result.issue("error", "simulation_flag_missing", source, row=index, field="simulated")
        # Lab07 never has real provider usage or billable monetary cost.
        for key in ("prompt_tokens", "completion_tokens", "total_tokens",
                    "estimated_cost_usd", "latency_ms"):
            if row[key] is not None:
                result.issue("error", "simulation_has_provider_measurement", source, row=index, field=key)
        attempts, retries, limits = (
            row["total_attempts"], row["retry_count"], row["rate_limit_response_count"]
        )
        if attempts is not None and attempts < 1:
            result.issue("error", "invalid_attempt_count", source, row=index, field="total_attempts")
        if attempts is not None and retries is not None and retries > max(0, attempts - 1):
            result.issue("error", "retry_count_exceeds_attempts", source, row=index, field="retry_count")
        if attempts is not None and limits is not None and limits > attempts:
            result.issue("error", "rate_limits_exceed_attempts", source, row=index, field="rate_limit_response_count")


def validate_run(results_dir: Path, *, experiment: str, run_id: str) -> ValidationResult:
    """Validate one allowlisted saved experiment run; never change its files.

    Version 1 denotes the normalized metric contract, *not* a universal raw
    input schema. Unversioned runs are admitted via the legacy adapter, with
    an explicit warning. Unsupported declared versions fail closed.
    """
    if experiment not in EXPERIMENT_SOURCES:
        raise ValueError("Unknown experiment; must be a registered experiment family")
    if not _safe_run_id(run_id):
        raise ValueError("Invalid run ID: must be a single directory name")
    root = Path(results_dir)
    if root.is_symlink():
        raise ValueError("Results root must not be a symbolic link")
    run_dir = root / experiment / run_id
    if (root / experiment).is_symlink() or run_dir.is_symlink():
        raise ValueError("Symlinked experiment/run directories are not allowed")
    if not run_dir.is_dir():
        raise ValueError("Saved experiment run does not exist")

    result = ValidationResult(experiment=experiment, run_id=run_id)
    manifest = _manifest_version(run_dir, result)
    if result.compatibility in ("unsupported", "invalid_manifest"):
        return result

    seen_stems: set[str] = set()
    for name in EXPERIMENT_SOURCES[experiment]:
        stem = Path(name).stem
        if stem in seen_stems:
            continue
        path = run_dir / name
        if not path.exists() and not path.is_symlink():
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                raise ValueError("File size exceeds permitted bound")
            rows = _read_rows(path)
        except (OSError, ValueError, UnicodeError):
            result.issue("error", "invalid_measurement_file", name)
            continue
        seen_stems.add(stem)
        result.file_count += 1
        for index, raw in enumerate(rows, 1):
            result.record_count += 1
            _validate_raw(raw, source=name, index=index, result=result)
            normalized = _normalize(
                raw, lab=experiment.split("_", 1)[0], experiment=experiment,
                run_id=run_id, source=name, manifest=manifest,
            )
            _validate_normalized(normalized, source=name, index=index, result=result)

    if result.file_count == 0:
        result.issue("error", "no_measurement_files", experiment)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate a saved lab run against normalized v1; offline and read-only."
    )
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--experiment", required=True, choices=tuple(EXPERIMENT_SOURCES))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--strict", action="store_true", help="Treat warnings as nonzero exit status.")
    parser.add_argument("--format", choices=("text", "json"), default="text")
    parser.add_argument("--output", type=Path, help="Save report outside the selected run folder.")
    args = parser.parse_args(argv)
    try:
        result = validate_run(args.results_dir, experiment=args.experiment, run_id=args.run_id)
    except ValueError as exc:
        parser.error(str(exc))

    report = result.to_report()
    if args.format == "json":
        rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    else:
        lines = [
            f"Experiment: {report['experiment']} | run: {report['run_id']}",
            f"Normalized contract: v{NORMALIZED_CONTRACT_VERSION} | compatibility: {report['compatibility']}",
            f"Checked: {report['files_checked']} file(s), {report['records_checked']} record(s)",
            f"Diagnostics: {report['errors']} errors, {report['warnings']} warnings",
        ]
        lines += [
            f"{item.severity}: {item.code} [{item.source}]"
            + (f" row {item.row}" if item.row is not None else "")
            + (f" field {item.field}" if item.field is not None else "")
            for item in result.diagnostics
        ]
        if report["diagnostics_truncated"]:
            lines.append(f"{report['diagnostics_truncated']} additional diagnostics truncated")
        rendered = "\n".join(lines) + "\n"

    if args.output is None:
        sys.stdout.write(rendered)
    else:
        destination = args.output.resolve()
        run_path = (args.results_dir / args.experiment / args.run_id).resolve()
        if destination == run_path or destination.is_relative_to(run_path):
            parser.error("Output must not overwrite or modify the selected saved experiment run")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered, encoding="utf-8")
        print(f"Saved validation report: {destination}")
    return int(not report["valid"] or (args.strict and report["warnings"] > 0))


if __name__ == "__main__":
    raise SystemExit(main())
