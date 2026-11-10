"""Dashboard label and comparison boundaries are deterministic and offline."""

from __future__ import annotations

import pytest

from lab_core.dashboard_view import (
    LAB_CATALOG,
    lab_measurement_counts,
    scenario_label,
    select_simulation_comparison,
    simulation_chart_rows,
    strategy_label,
)


def _sample_rows():
    return [
        {
            "run_id": "run-a", "scenario": "transient_failures",
            "strategy": "rate_limit_aware",
            "success_rate_percent": 100.0, "total_retries": 1,
            "total_429_responses": 0, "total_virtual_wait_seconds": 1.5,
        },
        {
            "run_id": "run-a", "scenario": "transient_failures",
            "strategy": "no_backoff",
            "success_rate_percent": 0.0, "total_retries": 0,
            "total_429_responses": 3, "total_virtual_wait_seconds": 0.0,
        },
        {
            "run_id": "run-b", "scenario": "transient_failures",
            "strategy": "exponential_backoff",
            "success_rate_percent": 50.0, "total_retries": 99,
            "total_429_responses": 9, "total_virtual_wait_seconds": 9.0,
        },
        {
            "run_id": "run-a", "scenario": "shared_contention",
            "strategy": "exponential_backoff_with_jitter",
            "success_rate_percent": 25.0, "total_retries": 40,
            "total_429_responses": 8, "total_virtual_wait_seconds": 18.0,
        },
    ]


def test_strategy_and_scenario_names_are_readable():
    assert strategy_label("no_backoff") == "No backoff"
    assert strategy_label("exponential_backoff") == "Exponential"
    assert strategy_label("exponential_backoff_with_jitter") == "Exponential + jitter"
    assert strategy_label("rate_limit_aware") == "Rate-limit aware"
    assert scenario_label("transient_failures") == "Transient failures"
    assert scenario_label("shared_contention") == "Shared contention"
    assert strategy_label("custom_policy") == "Custom policy"


def test_comparison_does_not_mix_runs_or_scenarios():
    rows = select_simulation_comparison(
        _sample_rows(), run_id="run-a", scenario="transient_failures"
    )
    assert [row["strategy"] for row in rows] == ["no_backoff", "rate_limit_aware"]
    assert all(row["run_id"] == "run-a" for row in rows)
    assert all(row["scenario"] == "transient_failures" for row in rows)
    assert select_simulation_comparison(
        _sample_rows(), run_id="unknown", scenario="transient_failures"
    ) == []


def test_chart_labels_are_short_and_preserve_zero():
    selected = select_simulation_comparison(
        _sample_rows(), run_id="run-a", scenario="transient_failures"
    )
    rows = simulation_chart_rows(selected, "success_rate_percent", "Success (%)")
    assert rows == [
        {"Strategy": "No backoff", "Success (%)": 0.0},
        {"Strategy": "Rate-limit aware", "Success (%)": 100.0},
    ]
    assert all("run-" not in row["Strategy"] for row in rows)


def test_missing_metrics_are_not_replaced_by_zero():
    rows = [{
        "strategy": "no_backoff", "total_retries": None,
        "success_rate_percent": 0,
    }]
    assert simulation_chart_rows(rows, "total_retries", "Retries") == []
    assert simulation_chart_rows(rows, "success_rate_percent", "Success (%)") == [
        {"Strategy": "No backoff", "Success (%)": 0}
    ]


def test_chart_field_allowlist_prevents_unintended_columns():
    with pytest.raises(ValueError, match="Unsupported"):
        simulation_chart_rows(_sample_rows(), "prompt_text", "Text")


def test_catalog_always_lists_all_ten_labs_even_without_results():
    assert LAB_CATALOG == tuple(f"lab{n:02d}" for n in range(1, 11))
    counts = lab_measurement_counts([])
    assert tuple(counts) == LAB_CATALOG
    assert all(value == 0 for value in counts.values())


def test_lab01_subexperiments_count_under_one_lab_and_other_labs_stay_visible():
    counts = lab_measurement_counts([
        {"lab": "lab01", "experiment": "lab01_effort"},
        {"lab": "lab01", "experiment": "lab01_factorial"},
        {"lab": "lab07", "experiment": "lab07_rate_limiting"},
        {"lab": "unexpected"},
    ])
    assert counts["lab01"] == 2
    assert counts["lab07"] == 1
    assert counts["lab02"] == 0
    assert len(counts) == 10
