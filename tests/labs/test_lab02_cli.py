"""Offline, mocked end-to-end tests for the `lab02` CLI: dry-run safety,
live execution (fake Groq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, and the
`lab02 backfill` command. No real Groq API calls anywhere.
"""

from __future__ import annotations

import json

import pytest

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


def _toml(tmp_path, live=True, **lab02_overrides) -> str:
    lab02_lines = "\n".join(f"{k} = {v if not isinstance(v, bool) else str(v).lower()}" for k, v in lab02_overrides.items())
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

[lab02]
{lab02_lines}
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

    config_path = _write_config(tmp_path, repetitions=2, multi_turn_turns=4, max_api_calls=10)
    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab02_context_growth").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1


def test_without_live_flag_no_api_call_is_made(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("must never construct a real client without --live")

    config_path = _write_config(tmp_path, live=False, repetitions=1, multi_turn_turns=1, max_api_calls=10)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab02", "minimal_context", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False


def test_live_run_writes_measurements_and_charts_in_one_command(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, multi_turn_turns=2, max_api_calls=10)
    responses = [make_response(content="4", prompt_tokens=100, completion_tokens=5) for _ in range(3)] + [
        make_response(content="fact", prompt_tokens=50, completion_tokens=5) for _ in range(2)
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--live", "--experiment-run-id", "cli-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab02_context_growth" / "cli-test"
    assert (run_dir / "context_measurements.json").exists()
    assert (run_dir / "multi_turn_measurements.json").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "analysis_report.md").exists()
    assert (run_dir / "condition_summary.csv").exists()

    charts_dir = run_dir / "charts"
    assert charts_dir.is_dir()
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 8
    for png in png_files:
        assert png.stat().st_size > 0

    with open(run_dir / "context_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 3
    assert all(r["answer_correct"] is True for r in records)


def test_single_condition_mode_runs_only_that_condition(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=2, multi_turn_turns=4, max_api_calls=10)
    responses = [make_response(content="4", prompt_tokens=1200, completion_tokens=5) for _ in range(2)]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab02", "irrelevant_context", "--live", "--experiment-run-id", "single-cond", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab02_context_growth" / "single-cond"
    with open(run_dir / "context_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
    assert all(r["condition_id"] == "irrelevant_context" for r in records)
    with open(run_dir / "multi_turn_measurements.json", encoding="utf-8") as f:
        assert json.load(f) == []


def test_cli_override_exceeding_budget_is_refused_with_zero_api_calls(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("budget-exceeding override must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, repetitions=2, multi_turn_turns=4, max_api_calls=10)
    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--live", "--repetitions", "5", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    assert not (tmp_path / "results" / "lab02_context_growth").exists() or not any(
        (tmp_path / "results" / "lab02_context_growth").glob("*/context_measurements.json")
    )


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, multi_turn_turns=1, max_api_calls=10)

    responses = [make_response(content="4", prompt_tokens=100, completion_tokens=5) for _ in range(3)] + [
        make_response(content="fact", prompt_tokens=50, completion_tokens=5)
    ]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("resuming a used run_id must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    args2 = parser.parse_args(["lab02", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)

    assert client_constructed["called"] is False


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, repetitions=1, multi_turn_turns=1, max_api_calls=10)
    responses = [make_response(content="4", prompt_tokens=100, completion_tokens=5) for _ in range(3)] + [
        make_response(content="fact", prompt_tokens=50, completion_tokens=5)
    ]
    _patch_groq(monkeypatch, responses)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab02_context_growth.charts.generate_context_growth_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab02_context_growth" / "chart-fail"
    assert (run_dir / "context_measurements.json").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_runs_without_live_or_api_key(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, repetitions=1, multi_turn_turns=1, max_api_calls=10)
    responses = [make_response(content="4", prompt_tokens=100, completion_tokens=5) for _ in range(3)] + [
        make_response(content="fact", prompt_tokens=50, completion_tokens=5)
    ]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab02", "all", "--live", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("backfill must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backfill_args = parser.parse_args(["lab02", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert client_constructed["called"] is False
