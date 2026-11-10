"""Presentation-only helpers for dashboard charts (no network or UI imports).

Make retry simulator labels readable without hiding experimental semantics.
Each chart must compare exactly one run and one scenario. Missing values
stay missing, including when other strategies reported zeros.
"""

from __future__ import annotations

from typing import Iterable

from lab_core.dashboard_data import EXPERIMENT_SOURCES


# Every implemented lab appears in the selector, even without saved results.
LAB_CATALOG = tuple(sorted({name.split("_", 1)[0] for name in EXPERIMENT_SOURCES}))


def lab_measurement_counts(records: Iterable[dict]) -> dict[str, int]:
    """Measurement coverage by lab; zero means no local results were found."""
    counts = {lab: 0 for lab in LAB_CATALOG}
    for record in records:
        lab = record.get("lab")
        if lab in counts:
            counts[lab] += 1
    return counts

STRATEGY_LABELS = {
    "no_backoff": "No backoff",
    "exponential_backoff": "Exponential",
    "exponential_backoff_with_jitter": "Exponential + jitter",
    "rate_limit_aware": "Rate-limit aware",
}

SCENARIO_LABELS = {
    "transient_failures": "Transient failures",
    "server_recovery_window": "Server recovery",
    "shared_contention": "Shared contention",
    "retry_after_variations": "Retry-After variants",
    "quota_pressure": "Quota pressure",
}

SIMULATION_CHART_FIELDS = frozenset({
    "success_rate_percent", "total_retries",
    "total_429_responses", "total_virtual_wait_seconds",
})


def strategy_label(value: str) -> str:
    return STRATEGY_LABELS.get(value, value.replace("_", " ").capitalize())


def scenario_label(value: str) -> str:
    return SCENARIO_LABELS.get(value, value.replace("_", " ").capitalize())


def select_simulation_comparison(
    summaries: Iterable[dict], *, run_id: str, scenario: str,
) -> list[dict]:
    """Select a single matched scenario and run; never mix groups in a chart."""
    rows = [
        r for r in summaries
        if r["run_id"] == run_id and r["scenario"] == scenario
    ]
    priority = {key: index for index, key in enumerate(STRATEGY_LABELS)}
    return sorted(rows, key=lambda r: (priority.get(r["strategy"], 999), r["strategy"]))


def simulation_chart_rows(
    summaries: Iterable[dict], field: str, axis_label: str,
) -> list[dict]:
    """Short axis labels; preserve zero and omit only missing observations."""
    if field not in SIMULATION_CHART_FIELDS:
        raise ValueError("Unsupported simulation comparison metric")
    return [
        {"Strategy": strategy_label(row["strategy"]), axis_label: row[field]}
        for row in summaries if row.get(field) is not None
    ]
