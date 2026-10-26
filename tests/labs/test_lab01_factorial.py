"""Legacy unit tests for the retired `lab01 factorial` full-factorial engine.

The `factorial` CLI execution mode itself was retired (see
labs.lab01_token_anatomy.cli._run_legacy_factorial_stub and
tests/labs/test_lab01_effort.py) in favor of the leaner, reasoning-effort-
only `effort` mode, to avoid the Groq rate-limit/quota pressure a 540-call
factorial sweep created. The underlying `labs.lab01_token_anatomy.factorial_*`
modules are kept, unmodified, as a library -- this file only exercises their
functions directly (never through the CLI, since that path now always
fails closed) so that retained code doesn't silently rot. Do not add new
CLI-level assertions here; new active-experiment coverage belongs in
test_lab01_effort.py.
"""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lab_core.config import Lab01FactorialConfig
from labs.lab01_token_anatomy.factorial_matrix import (
    FactorialMatrixError,
    build_conditions,
    build_conditions_from_config,
    build_execution_plan,
    build_plan_from_config,
)
from labs.lab01_token_anatomy.factorial_models import FactorialMeasurement, make_condition_id
from labs.lab01_token_anatomy.factorial_persistence import (
    ResumeConfigurationMismatch,
    append_measurement,
    build_initial_manifest,
    completed_pairs,
    compute_config_fingerprint,
    factorial_run_dir,
    read_existing_measurements,
    validate_resume_fingerprint,
    write_manifest,
)
from labs.lab01_token_anatomy.factorial_report import build_analysis_report_markdown
from labs.lab01_token_anatomy.factorial_runner import (
    Budget,
    evaluate_quality_heuristic,
    execute_planned_call,
)
from labs.lab01_token_anatomy.factorial_stats import (
    compute_descriptive_stats,
    compute_grouped_effect,
    compute_level_difference,
    summarize_all_conditions,
)

PROMPTS = ["short_prompt", "long_prompt"]
EFFORTS = ["low", "medium", "high"]
TEMPS = [0.5, 1.0, 1.5]
TOP_PS = [0.7, 0.9, 1.0]


def _factorial_config(**overrides) -> Lab01FactorialConfig:
    defaults = dict(
        enabled=True,
        prompts=tuple(PROMPTS),
        reasoning_efforts=tuple(EFFORTS),
        temperatures=tuple(TEMPS),
        top_p_values=tuple(TOP_PS),
        repetitions=10,
        max_completion_tokens=4096,
        randomize_execution_order=True,
        randomization_seed=42,
        use_paired_api_seed=False,
        max_api_calls=None,
        max_estimated_cost_usd=None,
    )
    defaults.update(overrides)
    return Lab01FactorialConfig(**defaults)


# --- Experiment Matrix ------------------------------------------------------


def test_full_matrix_produces_exactly_54_conditions():
    conditions = build_conditions(PROMPTS, EFFORTS, TEMPS, TOP_PS)
    assert len(conditions) == 54


def test_every_condition_is_unique():
    conditions = build_conditions(PROMPTS, EFFORTS, TEMPS, TOP_PS)
    assert len({c.condition_id for c in conditions}) == 54


def test_default_10_repetitions_yields_540_planned_calls():
    conditions = build_conditions(PROMPTS, EFFORTS, TEMPS, TOP_PS)
    plan = build_execution_plan(conditions, repetitions=10, randomize_execution_order=False, randomization_seed=1)
    assert len(plan) == 540


def test_parameter_filtering_reduces_matrix_correctly():
    conditions = build_conditions(["short_prompt"], ["low", "high"], TEMPS, TOP_PS)
    assert len(conditions) == 1 * 2 * 3 * 3 == 18


def test_unknown_prompt_type_is_rejected():
    with pytest.raises(FactorialMatrixError):
        build_conditions(["not_a_real_prompt"], EFFORTS, TEMPS, TOP_PS)


def test_condition_id_collision_is_rejected():
    # 0.701 and 0.699 both round to "0.70" at the 2-decimal precision
    # condition_id uses -- this must be caught, not silently merged.
    with pytest.raises(FactorialMatrixError, match="collision"):
        build_conditions(["short_prompt"], ["low"], [0.5], [0.701, 0.699])


def test_condition_id_is_stable_regardless_of_float_repr_noise():
    a = make_condition_id("short_prompt", "low", 0.5, 0.7)
    b = make_condition_id("short_prompt", "low", 0.50000000001, 0.7)
    assert a == b  # rounds to the same 2-decimal id


def test_build_plan_from_config_matches_expected_540_total():
    config = _factorial_config()
    conditions = build_conditions_from_config(config)
    plan = build_plan_from_config(config)
    assert len(conditions) == 54
    assert len(plan) == 540


# --- Reproducibility ---------------------------------------------------


def test_same_seed_produces_same_execution_order():
    conditions = build_conditions(PROMPTS, EFFORTS, TEMPS, TOP_PS)
    plan_a = build_execution_plan(conditions, repetitions=2, randomize_execution_order=True, randomization_seed=99)
    plan_b = build_execution_plan(conditions, repetitions=2, randomize_execution_order=True, randomization_seed=99)
    order_a = [(c.condition.condition_id, c.repetition_index) for c in plan_a]
    order_b = [(c.condition.condition_id, c.repetition_index) for c in plan_b]
    assert order_a == order_b


def test_different_seed_produces_a_different_execution_order():
    conditions = build_conditions(PROMPTS, EFFORTS, TEMPS, TOP_PS)
    plan_a = build_execution_plan(conditions, repetitions=2, randomize_execution_order=True, randomization_seed=1)
    plan_b = build_execution_plan(conditions, repetitions=2, randomize_execution_order=True, randomization_seed=2)
    order_a = [(c.condition.condition_id, c.repetition_index) for c in plan_a]
    order_b = [(c.condition.condition_id, c.repetition_index) for c in plan_b]
    assert order_a != order_b


def test_condition_identity_is_independent_of_execution_order():
    conditions = build_conditions(["short_prompt"], ["low"], [0.5], [0.7])
    plan_randomized = build_execution_plan(conditions, repetitions=5, randomize_execution_order=True, randomization_seed=1)
    plan_fixed = build_execution_plan(conditions, repetitions=5, randomize_execution_order=False, randomization_seed=1)
    assert {c.condition.condition_id for c in plan_randomized} == {c.condition.condition_id for c in plan_fixed} == {
        conditions[0].condition_id
    }


def test_paired_api_seed_shares_seed_across_conditions_for_the_same_repetition():
    conditions = build_conditions(["short_prompt", "long_prompt"], ["low"], [0.5], [0.7])
    plan = build_execution_plan(
        conditions, repetitions=3, randomize_execution_order=False, randomization_seed=10, use_paired_api_seed=True
    )
    by_repetition = {}
    for call in plan:
        by_repetition.setdefault(call.repetition_index, set()).add(call.api_seed)
    # every repetition_index maps to exactly one api_seed shared by both conditions
    assert all(len(seeds) == 1 for seeds in by_repetition.values())


def test_api_seed_is_none_when_paired_seed_disabled():
    conditions = build_conditions(["short_prompt"], ["low"], [0.5], [0.7])
    plan = build_execution_plan(conditions, repetitions=2, randomize_execution_order=False, randomization_seed=1)
    assert all(call.api_seed is None for call in plan)


def test_request_parameters_belong_to_the_correct_condition(fake_sync_client_factory, make_response):
    conditions = build_conditions(["short_prompt"], ["high"], [1.5], [0.7])
    plan = build_execution_plan(conditions, repetitions=1, randomize_execution_order=False, randomization_seed=1)
    client = fake_sync_client_factory([make_response(completion_tokens=10, reasoning_tokens=2)])

    from lab_core.config import RetryConfig

    retry_config = RetryConfig(max_attempts=1, base_delay_seconds=0.01, max_delay_seconds=0.01, jitter=False)
    execute_planned_call(client, retry_config, "openai/gpt-oss-20b", None, 4096, "run-1", plan[0])

    sent = client.completions.calls[0]
    assert sent["reasoning_effort"] == "high"
    assert sent["temperature"] == 1.5
    assert sent["top_p"] == 0.7
    assert sent["max_completion_tokens"] == 4096


# --- Observability -----------------------------------------------------


@pytest.fixture
def retry_config():
    from lab_core.config import RetryConfig

    return RetryConfig(max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)


def _one_call(client, retry_config, pricing=None, max_completion_tokens=4096, repetitions=1):
    conditions = build_conditions(["short_prompt"], ["low"], [0.5], [0.7])
    plan = build_execution_plan(conditions, repetitions=repetitions, randomize_execution_order=False, randomization_seed=1)
    return [
        execute_planned_call(client, retry_config, "openai/gpt-oss-20b", pricing, max_completion_tokens, "run-1", call)
        for call in plan
    ]


def test_reasoning_tokens_parsed_correctly(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(completion_tokens=41, reasoning_tokens=24)])
    [m] = _one_call(client, retry_config)
    assert m.reasoning_tokens == 24


def test_missing_reasoning_tokens_is_none(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(completion_tokens=41, reasoning_tokens=None)])
    [m] = _one_call(client, retry_config)
    assert m.reasoning_tokens is None
    assert m.non_reasoning_completion_tokens is None


def test_non_reasoning_token_calculation_is_correct(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(completion_tokens=41, reasoning_tokens=24)])
    [m] = _one_call(client, retry_config)
    assert m.non_reasoning_completion_tokens == 17


def test_finish_reason_recorded_correctly(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(finish_reason="length")])
    [m] = _one_call(client, retry_config)
    assert m.finish_reason == "length"


def test_latency_measurement_field_is_populated(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response()])
    [m] = _one_call(client, retry_config)
    assert m.total_latency_ms is not None and m.total_latency_ms >= 0


def test_estimated_cost_is_correct(fake_sync_client_factory, make_response, retry_config):
    from lab_core.config import ModelPricing

    pricing = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0, cached_input_per_1m_usd=0.5, price_verified_date="x")
    client = fake_sync_client_factory([make_response(prompt_tokens=1_000_000, completion_tokens=1_000_000)])
    [m] = _one_call(client, retry_config, pricing=pricing)
    assert m.estimated_cost_usd == pytest.approx(3.0)


def test_cost_does_not_double_count_reasoning_tokens(fake_sync_client_factory, make_response, retry_config):
    from lab_core.config import ModelPricing

    pricing = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0, cached_input_per_1m_usd=0.5, price_verified_date="x")
    with_r = fake_sync_client_factory([make_response(completion_tokens=500, reasoning_tokens=300)])
    without_r = fake_sync_client_factory([make_response(completion_tokens=500, reasoning_tokens=None)])
    [m1] = _one_call(with_r, retry_config, pricing=pricing)
    [m2] = _one_call(without_r, retry_config, pricing=pricing)
    assert m1.estimated_cost_usd == m2.estimated_cost_usd


def test_quality_heuristic_short_prompt_contains_paris():
    method, passed = evaluate_quality_heuristic("short_prompt", "The capital of France is Paris.")
    assert method == "heuristic_contains_paris"
    assert passed is True


def test_quality_heuristic_short_prompt_missing_paris():
    method, passed = evaluate_quality_heuristic("short_prompt", "I don't know.")
    assert passed is False


def test_quality_heuristic_long_prompt_not_evaluated():
    method, passed = evaluate_quality_heuristic("long_prompt", "a very long technical answer" * 10)
    assert method == "not_evaluated"
    assert passed is None


def test_raw_answer_text_is_never_persisted_in_measurement(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(content="SECRET_MARKER_TEXT")])
    [m] = _one_call(client, retry_config)
    serialized = json.dumps(m.__dict__)
    assert "SECRET_MARKER_TEXT" not in serialized
    assert m.final_answer_char_count == len("SECRET_MARKER_TEXT")


# --- Statistical Analysis -----------------------------------------------


def test_descriptive_stats_mean_median_correct():
    stats = compute_descriptive_stats([32, 38, 39, 39, 41])
    assert stats.mean == pytest.approx(37.8)
    assert stats.median == 39


def test_descriptive_stats_sample_standard_deviation_correct():
    stats = compute_descriptive_stats([2, 4, 4, 4, 5, 5, 7, 9])
    import statistics

    assert stats.sample_standard_deviation == pytest.approx(statistics.stdev([2, 4, 4, 4, 5, 5, 7, 9]))


def test_descriptive_stats_percentiles_correct():
    values = list(range(1, 101))  # 1..100
    stats = compute_descriptive_stats(values)
    assert stats.p95 is not None
    assert 94 <= stats.p95 <= 96


def test_descriptive_stats_missing_reasoning_measurements_not_zeroed():
    stats = compute_descriptive_stats([24, None, 30, None])
    assert stats.count == 2
    assert stats.count_missing == 2
    assert stats.mean == pytest.approx(27.0)


def test_descriptive_stats_single_observation_has_no_stdev_or_p95():
    stats = compute_descriptive_stats([42])
    assert stats.count == 1
    assert stats.sample_standard_deviation is None
    assert stats.p95 is None
    assert stats.mean == 42


def _make_measurement(**overrides) -> FactorialMeasurement:
    base = dict(
        experiment_run_id="run-1",
        condition_id="cond",
        execution_index=1,
        repetition_index=1,
        timestamp="2026-01-01T00:00:00Z",
        prompt_type="short_prompt",
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
        temperature=0.5,
        top_p=0.7,
        max_completion_tokens=4096,
        api_seed=None,
        succeeded=True,
        attempts_used=1,
        completion_tokens=40,
        reasoning_tokens=20,
        total_latency_ms=100.0,
        estimated_cost_usd=0.0001,
        finish_reason="stop",
    )
    base.update(overrides)
    return FactorialMeasurement(**base)


def test_interaction_comparisons_use_correct_groups():
    measurements = [
        _make_measurement(reasoning_effort="low", temperature=0.5, reasoning_tokens=10),
        _make_measurement(reasoning_effort="low", temperature=1.5, reasoning_tokens=20),
        _make_measurement(reasoning_effort="high", temperature=0.5, reasoning_tokens=30),
        _make_measurement(reasoning_effort="high", temperature=1.5, reasoning_tokens=50),
    ]
    grouped = compute_grouped_effect(measurements, ("reasoning_effort", "temperature"), "reasoning_tokens")
    assert grouped[("low", 0.5)].mean == 10
    assert grouped[("high", 1.5)].mean == 50

    by_temp_at_high = compute_grouped_effect(
        [m for m in measurements if m.reasoning_effort == "high"], ("temperature",), "reasoning_tokens"
    )
    diff = compute_level_difference(by_temp_at_high, 0.5, 1.5)
    assert diff == pytest.approx(20.0)


def test_failed_calls_excluded_from_success_statistics():
    measurements = [
        _make_measurement(succeeded=True, completion_tokens=40),
        _make_measurement(succeeded=False, completion_tokens=None, reasoning_tokens=None, error="boom", finish_reason=None),
    ]
    conditions_by_id = {"cond": 2}
    summaries = summarize_all_conditions(
        [SimpleNamespace(condition_id="cond", prompt_type="short_prompt", reasoning_effort="low", temperature=0.5, top_p=0.7)],
        conditions_by_id,
        measurements,
    )
    [summary] = summaries
    assert summary.successful_calls == 1
    assert summary.failed_calls == 1
    assert summary.completion_tokens.count == 1


def test_truncated_responses_reported_correctly():
    measurements = [
        _make_measurement(finish_reason="stop"),
        _make_measurement(finish_reason="length"),
        _make_measurement(finish_reason="length"),
    ]
    [summary] = summarize_all_conditions(
        [SimpleNamespace(condition_id="cond", prompt_type="short_prompt", reasoning_effort="low", temperature=0.5, top_p=0.7)],
        {"cond": 3},
        measurements,
    )
    assert summary.truncated_calls == 2
    assert summary.truncation_rate == pytest.approx(2 / 3)


def test_analysis_report_builds_without_crashing_on_mixed_success_and_failure():
    conditions = build_conditions(["short_prompt"], ["low", "high"], [0.5, 1.5], [0.7])
    measurements = [
        _make_measurement(condition_id=c.condition_id, reasoning_effort=c.reasoning_effort, temperature=c.temperature, top_p=c.top_p, prompt_type=c.prompt_type)
        for c in conditions
    ] + [_make_measurement(condition_id=conditions[0].condition_id, succeeded=False, error="boom", completion_tokens=None, reasoning_tokens=None, finish_reason=None)]
    summaries = summarize_all_conditions(conditions, {c.condition_id: 2 for c in conditions}, measurements)
    manifest = {
        "experiment_run_id": "run-1",
        "model": "openai/gpt-oss-20b",
        "experiment_configuration": {"repetitions": 2, "prompts": ["short_prompt"], "reasoning_efforts": ["low", "high"], "temperatures": [0.5, 1.5], "top_p_values": [0.7], "max_completion_tokens": 4096},
        "planned_conditions": len(conditions),
        "planned_calls": len(conditions) * 2,
        "configuration_fingerprint": "abc",
    }
    report = build_analysis_report_markdown(manifest, conditions, measurements, summaries)
    assert "# Lab01 Factorial Analysis Report" in report
    assert "Interaction Effects" in report


# --- Cost & Safety -------------------------------------------------------


def test_budget_refuses_to_exceed_max_api_calls():
    budget = Budget(max_api_calls=2, max_estimated_cost_usd=None, per_call_cost_ceiling=0.001)
    assert budget.can_make_another_call() is None
    budget.record_call(0.0001)
    assert budget.can_make_another_call() is None
    budget.record_call(0.0001)
    assert budget.can_make_another_call() is not None  # 2 calls made, cap is 2


def test_budget_refuses_to_exceed_cost_cap_using_worst_case_ceiling():
    budget = Budget(max_api_calls=None, max_estimated_cost_usd=0.01, per_call_cost_ceiling=0.006)
    assert budget.can_make_another_call() is None
    budget.record_call(0.006)
    # next call's worst case (0.006) + already spent (0.006) = 0.012 > 0.01 cap
    assert budget.can_make_another_call() is not None


# --- Persistence (direct function calls, not via CLI) ---------------------


def test_json_serialization_round_trip(tmp_path):
    run_dir = factorial_run_dir(tmp_path, "run-1")
    m = _make_measurement()
    append_measurement(run_dir, m, is_first_write=True)
    loaded = read_existing_measurements(run_dir)
    assert len(loaded) == 1
    assert loaded[0].condition_id == "cond"
    assert loaded[0].reasoning_tokens == 20


def test_csv_serialization_one_row_per_call(tmp_path):
    run_dir = factorial_run_dir(tmp_path, "run-1")
    for i in range(3):
        append_measurement(run_dir, _make_measurement(repetition_index=i + 1), is_first_write=(i == 0))
    import csv

    with open(run_dir / "raw_measurements.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3


def test_experiment_manifest_contains_required_fields():
    config = _factorial_config(repetitions=2)
    manifest = build_initial_manifest("run-1", config, "openai/gpt-oss-20b", planned_conditions=54, planned_calls=108)
    for key in [
        "experiment_run_id",
        "experiment_configuration",
        "model",
        "provider",
        "planned_conditions",
        "planned_calls",
        "execution_started_at",
        "configuration_fingerprint",
        "code_version_or_git_commit_if_available",
    ]:
        assert key in manifest
    assert "api_key" not in json.dumps(manifest)
    assert "GROQ_API_KEY" not in json.dumps(manifest)


def test_resume_does_not_repeat_completed_calls(tmp_path):
    run_dir = factorial_run_dir(tmp_path, "run-1")
    append_measurement(run_dir, _make_measurement(condition_id="cond", repetition_index=1), is_first_write=True)
    existing = read_existing_measurements(run_dir)
    done = completed_pairs(existing)
    assert ("cond", 1) in done
    assert ("cond", 2) not in done


def test_incompatible_configuration_resume_is_rejected(tmp_path):
    config_a = _factorial_config(repetitions=2)
    config_b = _factorial_config(repetitions=5)  # different -> different fingerprint
    run_dir = factorial_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config_a, "openai/gpt-oss-20b", 54, 108)
    write_manifest(run_dir, manifest)

    with pytest.raises(ResumeConfigurationMismatch):
        validate_resume_fingerprint(run_dir, config_b, "openai/gpt-oss-20b")


def test_compatible_configuration_resume_is_accepted(tmp_path):
    config = _factorial_config(repetitions=2)
    run_dir = factorial_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config, "openai/gpt-oss-20b", 54, 108)
    write_manifest(run_dir, manifest)
    validate_resume_fingerprint(run_dir, config, "openai/gpt-oss-20b")  # must not raise


def test_results_isolation_basic_mode_untouched(tmp_path):
    from lab_core.results_writer import write_json
    from lab_core.metrics import ExperimentResult, ExecutionMetadata, TokenCostMetrics, PerformanceMetrics

    basic_path = tmp_path / "lab01_token_anatomy.json"
    write_json(
        [
            ExperimentResult(
                metadata=ExecutionMetadata(
                    experiment_id="lab01_short_prompt",
                    run_id="r1",
                    model_id="openai/gpt-oss-20b",
                    execution_mode="sync",
                    timestamp="2026-01-01T00:00:00Z",
                ),
                tokens_and_cost=TokenCostMetrics(prompt_tokens=78, completion_tokens=40),
                performance=PerformanceMetrics(),
            )
        ],
        basic_path,
    )
    run_dir = factorial_run_dir(tmp_path, "run-1")
    append_measurement(run_dir, _make_measurement(), is_first_write=True)

    assert basic_path.exists()
    assert json.loads(basic_path.read_text(encoding="utf-8"))[0]["metadata"]["run_id"] == "r1"
    assert (run_dir / "raw_measurements.json").exists()


def test_config_fingerprint_is_stable_for_equivalent_config():
    config_a = _factorial_config()
    config_b = _factorial_config()
    assert compute_config_fingerprint(config_a, "openai/gpt-oss-20b") == compute_config_fingerprint(config_b, "openai/gpt-oss-20b")


def test_config_fingerprint_changes_when_matrix_changes():
    config_a = _factorial_config()
    config_b = replace(config_a, temperatures=(0.5, 1.0))
    assert compute_config_fingerprint(config_a, "openai/gpt-oss-20b") != compute_config_fingerprint(config_b, "openai/gpt-oss-20b")
