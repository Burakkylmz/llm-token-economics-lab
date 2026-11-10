"""Offline, mocked end-to-end tests for the `lab03` CLI: dry-run safety,
live execution (fake Groq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, and the
`lab03 backfill` command. No real Groq API calls anywhere.
"""

from __future__ import annotations

import json

from lab_core.cli import build_parser


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


def _toml(tmp_path, live=True, **lab03_overrides) -> str:
    lab03_lines = "\n".join(f"{k} = {v if not isinstance(v, bool) else str(v).lower()}" for k, v in lab03_overrides.items())
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

[lab02]
enabled = true

[lab05]
small_model = "openai/gpt-oss-20b"
large_model = "openai/gpt-oss-20b"

[lab09]
batch_completion_window = "24h"

[lab03]
caching_capable_models = ["openai/gpt-oss-20b"]
{lab03_lines}
"""


def _write_config(tmp_path, **kwargs):
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def test_dry_run_never_calls_api_and_writes_plan(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, repetitions=3, max_api_calls=8)
    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab03_prompt_caching").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 6  # header + 6 planned calls


def test_without_live_flag_no_api_call_is_made(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("must never construct a real client without --live")

    config_path = _write_config(tmp_path, live=False, repetitions=1, max_api_calls=8)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab03", "repeated_prefix", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False


def test_live_run_writes_measurements_and_charts_in_one_command(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=1500),
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--live", "--experiment-run-id", "cli-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab03_prompt_caching" / "cli-test"
    assert (run_dir / "cache_measurements.json").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "analysis_report.md").exists()
    assert (run_dir / "condition_summary.csv").exists()

    charts_dir = run_dir / "charts"
    assert charts_dir.is_dir()
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9
    for png in png_files:
        assert png.stat().st_size > 0

    with open(run_dir / "cache_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
    assert all(r["answer_correct"] is True for r in records)
    by_condition = {r["condition_id"]: r for r in records}
    assert by_condition["repeated_prefix"]["cached_tokens"] == 1500
    assert by_condition["cache_unfriendly"]["cached_tokens"] == 0


def test_single_condition_mode_runs_only_that_condition(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=2, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=1800),
        make_response(content="2", prompt_tokens=2000, completion_tokens=5, cached_tokens=1800),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab03", "repeated_prefix", "--live", "--experiment-run-id", "single-cond", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab03_prompt_caching" / "single-cond"
    with open(run_dir / "cache_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
    assert all(r["condition_id"] == "repeated_prefix" for r in records)


def test_cli_override_exceeding_budget_is_refused_with_zero_api_calls(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("budget-exceeding override must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, repetitions=3, max_api_calls=8)
    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--live", "--repetitions", "5", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    assert not (tmp_path / "results" / "lab03_prompt_caching").exists() or not any(
        (tmp_path / "results" / "lab03_prompt_caching").glob("*/cache_measurements.json")
    )


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
    ]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("resuming a used run_id must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    args2 = parser.parse_args(["lab03", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)

    assert client_constructed["called"] is False


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, repetitions=1, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
    ]
    _patch_groq(monkeypatch, responses)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab03_prompt_caching.charts.generate_cache_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab03_prompt_caching" / "chart-fail"
    assert (run_dir / "cache_measurements.json").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_cache_probe_plans_three_calls_by_default(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, repetitions=3, max_api_calls=8)
    parser = build_parser()
    args = parser.parse_args(["lab03", "cache_probe", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab03_prompt_caching").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 3


def test_cache_probe_live_run_writes_diagnostics_and_verification(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, repetitions=3, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=1454, completion_tokens=5, cached_tokens=None),
        make_response(content="5", prompt_tokens=1454, completion_tokens=5, cached_tokens=None),
        make_response(content="5", prompt_tokens=1454, completion_tokens=5, cached_tokens=0),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab03", "cache_probe", "--live", "--experiment-run-id", "probe-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab03_prompt_caching" / "probe-test"
    assert (run_dir / "cache_measurements.json").exists()
    assert (run_dir / "cache_diagnostics.json").exists()

    with open(run_dir / "cache_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 3
    assert len({r["request_hash"] for r in records}) == 1  # all three probe requests were byte-identical

    with open(run_dir / "cache_diagnostics.json", encoding="utf-8") as f:
        diagnostics = json.load(f)
    assert len(diagnostics) == 3
    expected_keys = {
        "provider_model", "sdk_version", "request_number", "condition_id", "succeeded",
        "prompt_tokens", "prompt_tokens_details_present", "cached_tokens_field_present",
        "cached_tokens", "cache_measurement_status", "raw_usage_field_names", "request_hash",
    }
    assert all(set(d.keys()) == expected_keys for d in diagnostics)  # bounded, no stray content fields

    out = capsys.readouterr().out
    assert "Cache Verification: NOT OBSERVED" in out
    assert "Prefix Stability Verification" in out


def test_cache_probe_observes_a_hit_when_the_mocked_response_reports_one(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, cache_probe_calls=2, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=1454, completion_tokens=5, cached_tokens=0),
        make_response(content="5", prompt_tokens=1454, completion_tokens=5, cached_tokens=1200),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab03", "cache_probe", "--live", "--experiment-run-id", "probe-hit", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "Cache Verification: OBSERVED" in out


def test_backfill_command_runs_without_live_or_api_key(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, max_api_calls=8)
    responses = [
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
        make_response(content="5", prompt_tokens=2000, completion_tokens=5, cached_tokens=0),
    ]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab03", "all", "--live", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("backfill must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backfill_args = parser.parse_args(["lab03", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert client_constructed["called"] is False
