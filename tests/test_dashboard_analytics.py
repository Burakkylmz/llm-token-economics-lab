"""Regression tests for read-only dashboard analytics selection and plotting.

Keep all analytic computations independent of Streamlit and provider SDKs.
"""

from __future__ import annotations

import pytest

from lab_core.dashboard_analytics import (
    GROUP_ID_FIELDS,
    METRIC_LABELS,
    available_runs,
    available_scenarios,
    build_scoped_benchmark,
    chart_rows,
    compare_selected,
    comparison_rows,
    group_id,
    groups_for_context,
    matching_contexts,
)


def record(**overrides):
    row = {
        "lab": "lab07", "experiment": "lab07_rate_limiting",
        "run_id": "pilot", "source": "raw_measurements.json",
        "scenario": "transient_failures", "model": "Simulation (no LLM)",
        "condition": "no_backoff", "cost_basis": "Unavailable",
        "simulated": True, "succeeded": True, "answer_correct": None,
        "retry_count": 1, "rate_limit_response_count": 1,
        "total_scheduled_wait_seconds": 0.0, "total_tokens": None,
        "estimated_cost_usd": None, "latency_ms": None,
    }
    return {**row, **overrides}


def data():
    return [
        record(succeeded=True, retry_count=1),
        record(succeeded=False, retry_count=2),
        record(condition="exponential_backoff", succeeded=True, retry_count=1,
               total_scheduled_wait_seconds=3.0),
        record(condition="exponential_backoff", succeeded=True, retry_count=1,
               total_scheduled_wait_seconds=5.0),
    ]


def test_run_and_scenario_discovery_does_not_mix_legacy_or_other_runs():
    rows = data() + [
        record(run_id="later", scenario="quota_pressure"),
        record(experiment="legacy", run_id="legacy"),
        record(lab="lab05", experiment="lab05_model_routing", run_id="saved",
               simulated=False),
    ]
    assert available_runs(rows) == [
        ("lab05", "saved"), ("lab07", "later"), ("lab07", "pilot"),
    ]
    assert available_scenarios(rows, lab="lab07", run_id="pilot") == ["transient_failures"]
    assert available_scenarios(rows, lab="lab07", run_id="later") == ["quota_pressure"]


def test_scoped_comparison_keeps_one_run_scenario_and_observed_denominators():
    rows = data() + [
        record(run_id="other", retry_count=900),
        record(scenario="quota_pressure", retry_count=900),
    ]
    report = build_scoped_benchmark(
        rows, lab="lab07", run_id="pilot", scenario="transient_failures",
    )
    assert report["measurement_count"] == 4
    assert report["group_count"] == 2
    contexts = matching_contexts(report)
    assert len(contexts) == 1
    group_options = groups_for_context(report, contexts[0])
    assert len(group_options) == 2
    assert all(len(option) == len(GROUP_ID_FIELDS) for option in group_options)
    assert {option[4] for option in group_options} == {
        "no_backoff", "exponential_backoff",
    }
    left = next(x for x in group_options if x[4] == "no_backoff")
    right = next(x for x in group_options if x[4] == "exponential_backoff")
    compared = compare_selected(report, left, right)
    assert compared["metrics"]["success_rate"]["percentage_point_delta"] == 50
    assert compared["metrics"]["retry_count"]["absolute_delta"] == -0.5
    assert compared["metrics"]["retry_count"]["baseline_observed"] == 2
    assert compared["metrics"]["retry_count"]["candidate_observed"] == 2
    assert compared["metrics"]["total_scheduled_wait_seconds"]["absolute_delta"] == 4
    assert compared["metrics"]["total_scheduled_wait_seconds"]["relative_delta_percent"] is None

    table = comparison_rows(compared)
    assert len(table) == len(METRIC_LABELS)
    by_metric = {row["Metric"]: row for row in table}
    assert by_metric["Client / request success"]["Delta"] == 50
    assert by_metric["Client / request success"]["Delta unit"] == "pp"
    assert by_metric["Mean simulated retries"]["Delta"] == -0.5
    assert by_metric["Mean simulated retries"]["Baseline n"] == 2
    assert by_metric["Mean estimated cost (USD)"]["Baseline"] is None

    values = chart_rows(compared, "total_scheduled_wait_seconds")
    assert values == [
        {"Group": "Baseline", "Observed value": 0.0, "Observed n": 2},
        {"Group": "Candidate", "Observed value": 4.0, "Observed n": 2},
    ]
    assert chart_rows(compared, "estimated_cost_usd") == []


def test_match_context_requires_two_distinct_conditions():
    rows = data()[:2]
    report = build_scoped_benchmark(
        rows, lab="lab07", run_id="pilot", scenario="transient_failures",
    )
    assert matching_contexts(report) == []
    with pytest.raises(ValueError, match="specific scenario"):
        build_scoped_benchmark(rows, lab="lab07", run_id="pilot", scenario="")
    with pytest.raises(ValueError, match="Invalid comparison context"):
        groups_for_context(report, ("only_one",))


def test_separate_sources_and_cost_basis_never_become_matched_context():
    rows = [
        record(condition="baseline", source="a.json", cost_basis="Estimated"),
        record(condition="candidate", source="b.json", cost_basis="Estimated"),
        record(condition="candidate", source="a.json", cost_basis="Batch discounted estimate"),
    ]
    report = build_scoped_benchmark(
        rows, lab="lab07", run_id="pilot", scenario="transient_failures",
    )
    assert matching_contexts(report) == []


def test_selector_rejects_unknown_duplicate_and_unmatched_contexts():
    report = build_scoped_benchmark(
        data(), lab="lab07", run_id="pilot", scenario="transient_failures",
    )
    groups = report["groups"]
    left, right = [group_id(g) for g in groups]
    with pytest.raises(ValueError, match="must be distinct"):
        compare_selected(report, left, left)
    with pytest.raises(ValueError, match="not in the selected saved run"):
        compare_selected(report, left, ("missing",))
    report["groups"][1]["source"] = "another_source.json"
    new_right = group_id(report["groups"][1])
    with pytest.raises(ValueError, match="source differs"):
        compare_selected(report, left, new_right)


def test_exploratory_model_difference_requires_explicit_acknowledgement():
    rows = [
        record(lab="lab05", experiment="lab05_model_routing",
               source="routing_measurements.json", simulated=False,
               model="small", condition="rule", cost_basis="Estimated",
               total_tokens=10, latency_ms=100, estimated_cost_usd=0.01),
        record(lab="lab05", experiment="lab05_model_routing",
               source="routing_measurements.json", simulated=False,
               model="large", condition="cascade", cost_basis="Estimated",
               total_tokens=20, latency_ms=200, estimated_cost_usd=0.02),
    ]
    report = build_scoped_benchmark(rows, lab="lab05", run_id="pilot",
                                    scenario="transient_failures")
    contexts = matching_contexts(report)
    assert len(contexts) == 1
    left, right = groups_for_context(report, contexts[0])
    with pytest.raises(ValueError, match="Model identifiers differ"):
        compare_selected(report, left, right)
    result = compare_selected(report, left, right, allow_model_difference=True)
    assert result["model_difference_acknowledged"]
    assert result["metrics"]["total_tokens"]["absolute_delta"] in (-10, 10)


def test_mixed_validity_preserves_coverage_and_chart_zero():
    rows = data() + [
        record(condition="exponential_backoff", retry_count=None,
               total_scheduled_wait_seconds=None),
    ]
    report = build_scoped_benchmark(
        rows, lab="lab07", run_id="pilot", scenario="transient_failures",
    )
    groups = {g["condition"]: group_id(g) for g in report["groups"]}
    result = compare_selected(report, groups["no_backoff"], groups["exponential_backoff"])
    retry = result["metrics"]["retry_count"]
    assert retry["baseline_observed"] == 2 and retry["candidate_observed"] == 2
    assert result["candidate"]["requests"] == 3
    with pytest.raises(ValueError, match="Unsupported"):
        chart_rows(result, "not_a_metric")


def test_no_direct_streamlit_dependency_in_pure_analytics_layer():
    import sys

    import lab_core.dashboard_analytics as analytics

    assert "streamlit" not in analytics.__dict__
    assert sys.modules.get("lab_core.dashboard_analytics") is analytics
