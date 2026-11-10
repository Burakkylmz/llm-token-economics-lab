from __future__ import annotations

import json

import pytest

from lab_core.cli import build_parser
from lab_core.config import Lab01EffortConfig
from labs.lab01_token_anatomy.effort_matrix import (
    EffortMatrixError,
    build_conditions,
    build_conditions_from_config,
    build_execution_plan,
    build_plan_from_config,
)
from labs.lab01_token_anatomy.effort_models import EffortMeasurement, make_effort_condition_id
from labs.lab01_token_anatomy.effort_persistence import (
    ResumeConfigurationMismatch,
    append_measurement,
    build_initial_manifest,
    compute_config_fingerprint,
    effort_run_dir,
    read_existing_measurements,
    validate_resume_fingerprint,
    write_manifest,
)
from labs.lab01_token_anatomy.effort_runner import (
    build_create_kwargs_for_condition,
    execute_planned_call,
)

DEFAULT_PROMPTS = ["short_prompt"]
EXTENDED_PROMPTS = ["short_prompt", "long_prompt"]
EFFORTS = ["low", "medium", "high"]


def _effort_config(**overrides) -> Lab01EffortConfig:
    defaults = dict(
        enabled=True,
        prompts=tuple(DEFAULT_PROMPTS),
        reasoning_efforts=tuple(EFFORTS),
        repetitions=2,
        max_completion_tokens=4096,
        randomize_execution_order=False,
        randomization_seed=42,
        use_paired_api_seed=False,
        max_api_calls=12,
        max_estimated_cost_usd=0.10,
    )
    defaults.update(overrides)
    return Lab01EffortConfig(**defaults)


# --- Experiment Matrix: default (6), minimal pilot (3), extended (12) ----


def test_default_matrix_produces_exactly_3_conditions():
    conditions = build_conditions(DEFAULT_PROMPTS, EFFORTS)
    assert len(conditions) == 3


def test_default_2_repetitions_yields_6_planned_calls():
    conditions = build_conditions(DEFAULT_PROMPTS, EFFORTS)
    plan = build_execution_plan(conditions, repetitions=2, randomize_execution_order=False, randomization_seed=1)
    assert len(plan) == 6


def test_minimal_pilot_yields_3_planned_calls():
    conditions = build_conditions(DEFAULT_PROMPTS, EFFORTS)
    plan = build_execution_plan(conditions, repetitions=1, randomize_execution_order=False, randomization_seed=1)
    assert len(plan) == 3


def test_extended_experiment_yields_6_conditions_and_12_planned_calls():
    conditions = build_conditions(EXTENDED_PROMPTS, EFFORTS)
    assert len(conditions) == 6
    plan = build_execution_plan(conditions, repetitions=2, randomize_execution_order=False, randomization_seed=1)
    assert len(plan) == 12


def test_conditions_have_no_temperature_or_top_p_attributes():
    [condition] = build_conditions(["short_prompt"], ["low"])
    assert not hasattr(condition, "temperature")
    assert not hasattr(condition, "top_p")


def test_every_condition_is_unique():
    conditions = build_conditions(EXTENDED_PROMPTS, EFFORTS)
    assert len({c.condition_id for c in conditions}) == 6


def test_unknown_prompt_type_is_rejected():
    with pytest.raises(EffortMatrixError):
        build_conditions(["not_a_real_prompt"], EFFORTS)


def test_condition_id_is_simple_and_deterministic():
    assert make_effort_condition_id("short_prompt", "low") == "short_prompt__low"


def test_build_plan_from_config_matches_default_6_total():
    config = _effort_config()
    conditions = build_conditions_from_config(config)
    plan = build_plan_from_config(config)
    assert len(conditions) == 3
    assert len(plan) == 6


# --- API request must never include temperature/top_p --------------------


def test_request_kwargs_never_include_temperature_or_top_p():
    kwargs = build_create_kwargs_for_condition("openai/gpt-oss-20b", "short_prompt", "low", 4096)
    assert "temperature" not in kwargs
    assert "top_p" not in kwargs
    assert kwargs["reasoning_effort"] == "low"
    assert kwargs["max_completion_tokens"] == 4096


def test_execute_planned_call_sends_reasoning_effort_but_not_temperature_or_top_p(
    fake_sync_client_factory, make_response
):
    from lab_core.config import RetryConfig

    conditions = build_conditions(["short_prompt"], ["high"])
    plan = build_execution_plan(conditions, repetitions=1, randomize_execution_order=False, randomization_seed=1)
    client = fake_sync_client_factory([make_response(completion_tokens=10, reasoning_tokens=2)])
    retry_config = RetryConfig(max_attempts=1, base_delay_seconds=0.01, max_delay_seconds=0.01, jitter=False)

    execute_planned_call(client, retry_config, "openai/gpt-oss-20b", None, 4096, "run-1", plan[0])

    sent = client.completions.calls[0]
    assert sent["reasoning_effort"] == "high"
    assert "temperature" not in sent
    assert "top_p" not in sent
    assert sent["max_completion_tokens"] == 4096


# --- Observability (reuses the same guarantees as the retired factorial engine) ---


@pytest.fixture
def retry_config():
    from lab_core.config import RetryConfig

    return RetryConfig(max_attempts=3, base_delay_seconds=0.001, max_delay_seconds=0.002, jitter=False)


def _one_call(client, retry_config, pricing=None, max_completion_tokens=4096):
    conditions = build_conditions(["short_prompt"], ["low"])
    plan = build_execution_plan(conditions, repetitions=1, randomize_execution_order=False, randomization_seed=1)
    return execute_planned_call(client, retry_config, "openai/gpt-oss-20b", pricing, max_completion_tokens, "run-1", plan[0])


def test_reasoning_tokens_parsed_correctly(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(completion_tokens=41, reasoning_tokens=24)])
    m = _one_call(client, retry_config)
    assert m.reasoning_tokens == 24
    assert m.non_reasoning_completion_tokens == 17


def test_missing_reasoning_metadata_is_none_not_zero(fake_sync_client_factory, make_response, retry_config):
    client = fake_sync_client_factory([make_response(completion_tokens=41, reasoning_tokens=None)])
    m = _one_call(client, retry_config)
    assert m.reasoning_tokens is None
    assert m.non_reasoning_completion_tokens is None


def test_cost_calculation_correct_and_no_double_counting(fake_sync_client_factory, make_response, retry_config):
    from lab_core.config import ModelPricing

    pricing = ModelPricing(input_per_1m_usd=1.0, output_per_1m_usd=2.0, cached_input_per_1m_usd=0.5, price_verified_date="x")
    with_r = fake_sync_client_factory([make_response(completion_tokens=500, reasoning_tokens=300)])
    without_r = fake_sync_client_factory([make_response(completion_tokens=500, reasoning_tokens=None)])
    m1 = _one_call(with_r, retry_config, pricing=pricing)
    m2 = _one_call(without_r, retry_config, pricing=pricing)
    # make_response's default prompt_tokens is 10 (see conftest.make_usage)
    assert m1.estimated_cost_usd == m2.estimated_cost_usd == pytest.approx(10 / 1_000_000 * 1.0 + 500 / 1_000_000 * 2.0)


# --- Statistical summary (single-factor) ----------------------------------


def _make_measurement(**overrides) -> EffortMeasurement:
    base = dict(
        experiment_run_id="run-1",
        condition_id="cond",
        execution_index=1,
        repetition_index=1,
        timestamp="2026-01-01T00:00:00Z",
        prompt_type="short_prompt",
        model="openai/gpt-oss-20b",
        reasoning_effort="low",
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
    return EffortMeasurement(**base)


def test_statistical_summary_mean_is_correct():
    from labs.lab01_token_anatomy.effort_stats import compute_descriptive_stats

    stats = compute_descriptive_stats([10, 20, 30])
    assert stats.mean == pytest.approx(20.0)


def test_single_repetition_pilot_standard_deviation_is_na_not_crash():
    from labs.lab01_token_anatomy.effort_stats import compute_descriptive_stats

    stats = compute_descriptive_stats([42])
    assert stats.sample_standard_deviation is None
    assert stats.mean == 42


def test_condition_summary_excludes_failed_calls_from_statistics():
    from labs.lab01_token_anatomy.effort_stats import summarize_all_effort_conditions
    from labs.lab01_token_anatomy.effort_models import EffortCondition

    measurements = [
        _make_measurement(succeeded=True, completion_tokens=40),
        _make_measurement(succeeded=False, completion_tokens=None, reasoning_tokens=None, error="boom", finish_reason=None),
    ]
    [summary] = summarize_all_effort_conditions(
        [EffortCondition(condition_id="cond", prompt_type="short_prompt", reasoning_effort="low")],
        {"cond": 2},
        measurements,
    )
    assert summary.successful_calls == 1
    assert summary.failed_calls == 1
    assert summary.completion_tokens.count == 1


# --- CLI overrides must be consistent across the whole chain --------------


def _toml(
    tmp_path,
    prompts=("short_prompt",),
    repetitions=2,
    max_api_calls=12,
    max_estimated_cost_usd=0.10,
    live=False,
) -> str:
    prompts_str = str(list(prompts))
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

[lab01.effort]
enabled = true
prompts = {prompts_str}
reasoning_efforts = ["low", "medium", "high"]
repetitions = {repetitions}
max_completion_tokens = 64
randomize_execution_order = false
randomization_seed = 1
use_paired_api_seed = false
max_api_calls = {max_api_calls}
max_estimated_cost_usd = {max_estimated_cost_usd}
"""


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
        def __init__(self, api_key=None, timeout=None, max_retries=None):
            self.api_key = api_key
            self.chat = _FakeChat(completions)

    _FakeClient.completions = completions
    return _FakeClient


def _patch_groq(monkeypatch, responses):
    fake_class = _make_fake_groq_class(responses)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", fake_class)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", fake_class)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return fake_class


def test_dry_run_minimal_pilot_plans_3_calls(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--dry-run", "--repetitions", "1", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "Unique Conditions: 3" in out
    assert "Planned API Calls: 3" in out
    assert "Repetitions: 1" in out


def test_dry_run_default_experiment_plans_6_calls(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=2, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--dry-run", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "Unique Conditions: 3" in out
    assert "Planned API Calls: 6" in out


def test_dry_run_extended_experiment_plans_12_calls_via_cli_override(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=2, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(
        [
            "lab01", "effort", "--dry-run",
            "--prompt", "short_prompt", "long_prompt",
            "--config", str(config_path),
        ]
    )
    args.func(args)

    out = capsys.readouterr().out
    assert "Unique Conditions: 6" in out
    assert "Planned API Calls: 12" in out


def test_dry_run_header_never_mentions_temperature_or_top_p_as_varying(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=2, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--dry-run", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "Temperature: provider default (not explicitly set)" in out
    assert "Top-p: provider default (not explicitly set)" in out
    assert "Temperature Levels" not in out
    assert "Top-p Levels" not in out


def test_dry_run_never_constructs_a_real_client(monkeypatch, tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=False), encoding="utf-8")

    def _fail_if_called(*a, **kw):
        raise AssertionError("dry-run must never construct a real Groq client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _fail_if_called)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _fail_if_called)

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--dry-run", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert (tmp_path / "results" / "lab01_effort").exists()


def test_cli_override_propagates_through_manifest_and_condition_summary(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=2, live=True), encoding="utf-8")
    fake_class = _patch_groq(monkeypatch, [make_response(completion_tokens=10) for _ in range(3)])

    parser = build_parser()
    args = parser.parse_args(
        [
            "lab01", "effort", "--live",
            "--repetitions", "1",
            "--experiment-run-id", "override-test",
            "--config", str(config_path),
        ]
    )
    args.func(args)

    assert len(fake_class.completions.calls) == 3  # 3 conditions x 1 repetition, never 2x more

    run_dir = tmp_path / "results" / "lab01_effort" / "override-test"
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["experiment_configuration"]["repetitions"] == 1
    assert manifest["planned_calls"] == 3

    summary = json.loads((run_dir / "effort_summary.json").read_text(encoding="utf-8"))
    assert summary["planned_calls"] == 3
    for condition_summary in summary["conditions"]:
        assert condition_summary["planned_calls"] == 1


def test_manifest_records_real_provider_model_prompt_and_effort(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    _patch_groq(monkeypatch, [make_response() for _ in range(3)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--live", "--experiment-run-id", "metadata-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab01_effort" / "metadata-test"
    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["model"] == "openai/gpt-oss-20b"
    assert manifest["provider"] == "groq"
    assert manifest["experiment_configuration"]["prompts"] == ["short_prompt"]
    assert manifest["experiment_configuration"]["reasoning_efforts"] == ["low", "medium", "high"]


def test_successful_live_run_auto_generates_charts(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    _patch_groq(monkeypatch, [make_response() for _ in range(3)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--live", "--experiment-run-id", "chart-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab01_effort" / "chart-test"
    charts_dir = run_dir / "charts"
    assert charts_dir.is_dir()
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 6
    for png in png_files:
        assert png.stat().st_size > 0
    # Auto-generated charts must never touch the measurement/report files.
    assert (run_dir / "raw_measurements.json").exists()
    assert (run_dir / "analysis_report.md").exists()


def test_resume_fingerprint_rejects_wrong_configuration(tmp_path):
    config_a = _effort_config(repetitions=2)
    config_b = _effort_config(repetitions=5)
    run_dir = effort_run_dir(tmp_path, "run-1")
    manifest = build_initial_manifest("run-1", config_a, "openai/gpt-oss-20b", 3, 6)
    write_manifest(run_dir, manifest)

    with pytest.raises(ResumeConfigurationMismatch):
        validate_resume_fingerprint(run_dir, config_b, "openai/gpt-oss-20b")


def test_resume_via_cli_does_not_repeat_completed_calls(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    fake_class = _patch_groq(monkeypatch, [make_response(completion_tokens=10) for _ in range(3)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--live", "--config", str(config_path), "--experiment-run-id", "resume-test"])
    args.func(args)
    assert len(fake_class.completions.calls) == 3

    args2 = parser.parse_args(["lab01", "effort", "--live", "--config", str(config_path), "--experiment-run-id", "resume-test"])
    args2.func(args2)
    assert len(fake_class.completions.calls) == 3  # unchanged -- no duplicate calls


# --- Rate limit / retry safety ---------------------------------------------


def test_rate_limit_429_is_retried_and_recorded_as_one_successful_measurement(monkeypatch, tmp_path, make_response):
    import httpx
    from groq import RateLimitError

    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={})
    rate_limit_error = RateLimitError("rate limited", response=response, body=None)

    responses = [
        rate_limit_error, make_response(completion_tokens=10),  # condition 1: retried once, then succeeds
        make_response(completion_tokens=11),  # condition 2
        make_response(completion_tokens=12),  # condition 3
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--live", "--config", str(config_path), "--experiment-run-id", "retry-test"])
    args.func(args)

    run_dir = tmp_path / "results" / "lab01_effort" / "retry-test"
    measurements = read_existing_measurements(run_dir)
    assert len(measurements) == 3  # no duplicate measurement from the retried attempt
    assert sum(1 for m in measurements if m.attempts_used == 2) == 1
    assert all(m.succeeded for m in measurements)


def test_daily_quota_exhaustion_does_not_retry_unboundedly(monkeypatch, tmp_path, make_response):
    # A persistent 429 (e.g. daily quota, not a transient blip) must give up
    # after retry_config.max_attempts, not retry forever.
    import httpx
    from groq import RateLimitError

    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(429, request=request, headers={})
    always_rate_limited = [RateLimitError("rate limited", response=response, body=None) for _ in range(10)]
    fake_class = _patch_groq(monkeypatch, always_rate_limited)

    parser = build_parser()
    args = parser.parse_args(
        [
            "lab01", "effort", "--live",
            "--prompt", "short_prompt",
            "--reasoning-effort", "low",
            "--repetitions", "1",
            "--config", str(config_path),
            "--experiment-run-id", "quota-test",
        ]
    )
    args.func(args)

    # retry_config.max_attempts=2 in this TOML -- exactly 2 attempts for the
    # single planned call, never more.
    assert len(fake_class.completions.calls) == 2

    run_dir = tmp_path / "results" / "lab01_effort" / "quota-test"
    measurements = read_existing_measurements(run_dir)
    assert len(measurements) == 1
    assert measurements[0].succeeded is False


def test_an_error_is_never_recorded_as_a_successful_measurement(monkeypatch, tmp_path):
    import httpx
    from groq import BadRequestError

    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")

    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(400, request=request, headers={})
    bad_request = BadRequestError("bad request", response=response, body=None)
    _patch_groq(monkeypatch, [bad_request])

    parser = build_parser()
    args = parser.parse_args(
        [
            "lab01", "effort", "--live",
            "--prompt", "short_prompt",
            "--reasoning-effort", "low",
            "--repetitions", "1",
            "--config", str(config_path),
            "--experiment-run-id", "error-test",
        ]
    )
    args.func(args)

    run_dir = tmp_path / "results" / "lab01_effort" / "error-test"
    [measurement] = read_existing_measurements(run_dir)
    assert measurement.succeeded is False
    assert measurement.completion_tokens is None


def test_request_budget_stops_before_exceeding_max_api_calls(monkeypatch, tmp_path, make_response):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=2, max_api_calls=2, live=True), encoding="utf-8")
    _patch_groq(monkeypatch, [make_response() for _ in range(6)])

    parser = build_parser()
    args = parser.parse_args(["lab01", "effort", "--live", "--config", str(config_path)])
    args.func(args)

    run_dirs = list((tmp_path / "results" / "lab01_effort").iterdir())
    run_dir = run_dirs[0]
    measurements = read_existing_measurements(run_dir)
    assert len(measurements) == 2  # stopped at the cap, not all 6

    manifest = json.loads((run_dir / "experiment_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stopped_reason"] is not None


# --- Legacy factorial command: always fails closed -------------------------


def test_legacy_factorial_command_always_fails_closed_regardless_of_flags(monkeypatch, tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=True), encoding="utf-8")
    monkeypatch.setenv("GROQ_API_KEY", "fake-key-not-real")

    def _fail_if_called(*a, **kw):
        raise AssertionError("the retired factorial command must never construct a Groq client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _fail_if_called)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _fail_if_called)

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--live", "--config", str(config_path)])
    args.func(args)  # must not raise, must not construct a client

    out = capsys.readouterr().out
    assert "retired" in out.lower()
    assert "effort" in out.lower()
    assert not (tmp_path / "results" / "lab01_factorial").exists()


def test_legacy_factorial_command_ignores_dry_run_and_other_flags_too(tmp_path, capsys):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "factorial", "--dry-run", "--config", str(config_path)])
    args.func(args)  # must not raise

    out = capsys.readouterr().out
    assert "retired" in out.lower()
    assert not (tmp_path / "results" / "lab01_factorial").exists()


# --- Regression: basic mode and other labs are untouched -------------------


def test_lab01_basic_mode_unaffected_by_effort_mode(tmp_path):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, repetitions=1, live=False), encoding="utf-8")

    parser = build_parser()
    args = parser.parse_args(["lab01", "short_prompt", "--config", str(config_path)])
    args.func(args)  # no --live: must not attempt any API call or crash

    assert not (tmp_path / "results" / "lab01_token_anatomy.json").exists()
