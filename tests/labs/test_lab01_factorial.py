from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from lab_core.cli import build_parser
from lab_core.config import ConfigError, Lab01FactorialConfig, load_config
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


def test_build_plan_from_config_matches_expected_540_total(monkeypatch=None):
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


def test_dry_run_builds_plan_without_any_client(monkeypatch, tmp_path):
    from labs.lab01_token_anatomy.factorial_cli import run_factorial

    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--dry-run", "--config", str(config_path)])

    def _fail_if_called(*a, **kw):
        raise AssertionError("dry-run must never construct a real Groq client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _fail_if_called)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _fail_if_called)
    args.func(args)  # must not raise

    plan_path = tmp_path / "results" / "lab01_factorial"
    assert plan_path.exists()


def test_default_live_safety_still_requires_live_flag(tmp_path):
    from labs.lab01_token_anatomy.factorial_cli import run_factorial

    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--config", str(config_path)])
    args.func(args)  # no --dry-run, no --live: must not attempt any API call or crash

    assert not (tmp_path / "results" / "lab01_factorial").exists()


# --- Persistence ----------------------------------------------------------


def _minimal_toml(tmp_path, repetitions=2, max_api_calls=None, max_estimated_cost_usd=None, live=False) -> str:
    extra_budget = ""
    if max_api_calls is not None:
        extra_budget += f"\nmax_api_calls = {max_api_calls}"
    if max_estimated_cost_usd is not None:
        extra_budget += f"\nmax_estimated_cost_usd = {max_estimated_cost_usd}"
    return f"""
[groq]
default_model = "openai/gpt-oss-20b"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = {"true" if live else "false"}
cost_warning_message = "warning"

[experiment]
default_repetitions = 2
results_dir = "{(tmp_path / "results").as_posix()}"

[retry]
max_attempts = 2
base_delay_seconds = 0.001
max_delay_seconds = 0.002
jitter = false

[concurrency]
max_concurrent_requests = 2

[pricing.models."openai/gpt-oss-20b"]
input_per_1m_usd = 0.075
output_per_1m_usd = 0.30
cached_input_per_1m_usd = 0.0375
price_verified_date = "2026-10-21"

[lab03]
caching_capable_models = ["openai/gpt-oss-20b"]

[lab05]
small_model = "openai/gpt-oss-20b"
large_model = "openai/gpt-oss-20b"

[lab09]
batch_completion_window = "24h"

[lab01.factorial]
enabled = true
prompts = ["short_prompt"]
reasoning_efforts = ["low"]
temperatures = [0.5]
top_p_values = [0.7]
repetitions = {repetitions}
max_completion_tokens = 64
randomize_execution_order = false
randomization_seed = 1
use_paired_api_seed = false{extra_budget}
"""


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


# --- Regression (Python 3.10 syntax compatibility smoke check) ------------


def test_config_fingerprint_is_stable_for_equivalent_config():
    config_a = _factorial_config()
    config_b = _factorial_config()
    assert compute_config_fingerprint(config_a, "openai/gpt-oss-20b") == compute_config_fingerprint(config_b, "openai/gpt-oss-20b")


def test_config_fingerprint_changes_when_matrix_changes():
    config_a = _factorial_config()
    config_b = replace(config_a, temperatures=(0.5, 1.0))
    assert compute_config_fingerprint(config_a, "openai/gpt-oss-20b") != compute_config_fingerprint(config_b, "openai/gpt-oss-20b")


# --- Full CLI integration: live execution, resume, budget-stop, retries ---
#
# These never touch the real network: `Groq`/`AsyncGroq` are monkeypatched
# at the exact names lab_core.lab_cli_support imports, so build_context's
# `Groq(api_key=...)` call returns a fake client whose `.chat.completions.
# create()` returns canned, pre-built response objects.


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        if not self._responses:
            raise AssertionError("fake client ran out of canned responses")
        next_response = self._responses.pop(0)
        if isinstance(next_response, Exception):
            raise next_response
        return next_response


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


def _make_fake_groq_class(responses):
    completions = _FakeCompletions(responses)

    class _FakeClient:
        def __init__(self, api_key=None):
            self.api_key = api_key
            self.chat = _FakeChat(completions)

    _FakeClient.completions = completions  # convenient test access
    return _FakeClient


def _patch_groq(monkeypatch, responses):
    fake_class = _make_fake_groq_class(responses)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", fake_class)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", fake_class)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return fake_class


def test_live_execution_writes_raw_measurements_and_summary_files(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    _patch_groq(monkeypatch, [make_response(completion_tokens=41, reasoning_tokens=24, finish_reason="stop")])

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--live", "--config", str(config_path)])
    args.func(args)

    run_dirs = list((tmp_path / "results" / "lab01_factorial").iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]
    for filename in ["experiment_manifest.json", "experiment_plan.csv", "raw_measurements.json", "raw_measurements.csv", "condition_summary.csv", "factorial_summary.json", "analysis_report.md"]:
        assert (run_dir / filename).exists(), filename

    measurements = read_existing_measurements(run_dir)
    assert len(measurements) == 1
    assert measurements[0].reasoning_tokens == 24
    assert measurements[0].succeeded is True


def test_resume_via_cli_does_not_repeat_completed_calls(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=2, live=True), encoding="utf-8")
    fake_class = _patch_groq(monkeypatch, [make_response(completion_tokens=10), make_response(completion_tokens=20)])

    parser = build_parser()
    args = parser.parse_args(
        ["lab01", "factorial", "--live", "--config", str(config_path), "--experiment-run-id", "resume-test"]
    )
    args.func(args)
    assert len(fake_class.completions.calls) == 2  # both repetitions ran

    run_dir = tmp_path / "results" / "lab01_factorial" / "resume-test"
    measurements_after_first_run = read_existing_measurements(run_dir)
    assert len(measurements_after_first_run) == 2

    # "Resuming" the same, already-fully-completed run must not make any new calls.
    args2 = parser.parse_args(
        ["lab01", "factorial", "--live", "--config", str(config_path), "--experiment-run-id", "resume-test"]
    )
    args2.func(args2)
    assert len(fake_class.completions.calls) == 2  # unchanged -- no duplicate calls
    measurements_after_second_run = read_existing_measurements(run_dir)
    assert len(measurements_after_second_run) == 2


def test_resume_with_changed_configuration_is_refused(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    _patch_groq(monkeypatch, [make_response()])

    parser = build_parser()
    args = parser.parse_args(
        ["lab01", "factorial", "--live", "--config", str(config_path), "--experiment-run-id", "mismatch-test"]
    )
    args.func(args)

    changed_config_path = tmp_path / "config_changed.toml"
    changed_config_path.write_text(_minimal_toml(tmp_path, repetitions=3, live=True), encoding="utf-8")
    args2 = parser.parse_args(
        ["lab01", "factorial", "--live", "--config", str(changed_config_path), "--experiment-run-id", "mismatch-test"]
    )
    args2.func(args2)  # must not raise -- the mismatch is reported as a printed message

    run_dir = tmp_path / "results" / "lab01_factorial" / "mismatch-test"
    # still only the one measurement from the first (valid) run -- the second,
    # mismatched attempt must not have appended anything.
    assert len(read_existing_measurements(run_dir)) == 1


def test_request_budget_stops_before_exceeding_max_api_calls(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        _minimal_toml(tmp_path, repetitions=5, max_api_calls=2, live=True), encoding="utf-8"
    )
    _patch_groq(monkeypatch, [make_response() for _ in range(5)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--live", "--config", str(config_path)])
    args.func(args)

    run_dirs = list((tmp_path / "results" / "lab01_factorial").iterdir())
    run_dir = run_dirs[0]
    measurements = read_existing_measurements(run_dir)
    assert len(measurements) == 2  # stopped at the max_api_calls cap, not all 5

    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stopped_reason"] is not None
    assert "max_api_calls" in manifest["stopped_reason"]


def test_retry_recovers_from_one_transient_failure_without_duplicate_measurement(monkeypatch, tmp_path, make_response):
    import httpx
    from groq import RateLimitError

    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={})
    rate_limit_error = RateLimitError("rate limited", response=response, body=None)

    _patch_groq(monkeypatch, [rate_limit_error, make_response(completion_tokens=15)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--live", "--config", str(config_path)])
    args.func(args)

    run_dirs = list((tmp_path / "results" / "lab01_factorial").iterdir())
    measurements = read_existing_measurements(run_dirs[0])
    # exactly ONE measurement, despite TWO underlying HTTP attempts
    assert len(measurements) == 1
    assert measurements[0].succeeded is True
    assert measurements[0].attempts_used == 2
    assert measurements[0].completion_tokens == 15


def test_non_retryable_error_is_recorded_as_a_failed_call_without_retrying_or_aborting_the_run(monkeypatch, tmp_path):
    # A single rejected parameter combination (e.g. an unsupported combo the
    # API itself refuses with 400) must show up as one explicit failed
    # measurement -- not crash the whole factorial campaign, and not be
    # silently retried as if it might succeed on attempt 2.
    import httpx
    from groq import BadRequestError

    config_path = tmp_path / "config.toml"
    config_path.write_text(_minimal_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, headers={})
    bad_request = BadRequestError("bad request", response=response, body=None)
    fake_class = _patch_groq(monkeypatch, [bad_request])

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--live", "--config", str(config_path)])
    args.func(args)  # must NOT raise -- the failure is recorded, not propagated

    assert len(fake_class.completions.calls) == 1  # no retry attempts for a non-retryable error

    run_dirs = list((tmp_path / "results" / "lab01_factorial").iterdir())
    measurements = read_existing_measurements(run_dirs[0])
    assert len(measurements) == 1
    assert measurements[0].succeeded is False
    assert measurements[0].attempts_used == 1
    assert "BadRequestError" in measurements[0].error
