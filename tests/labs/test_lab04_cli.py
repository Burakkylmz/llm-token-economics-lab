"""Offline, mocked end-to-end tests for the `lab04` CLI: dry-run safety,
live execution (fake Groq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, and the
`lab04 backfill` command. No real Groq API calls anywhere.
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


def _fmt_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, list):
        return "[" + ", ".join(_fmt_toml_value(x) for x in v) + "]"
    return str(v)


def _toml(tmp_path, live=True, **lab04_overrides) -> str:
    lab04_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab04_overrides.items())
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

[lab03]
caching_capable_models = ["openai/gpt-oss-20b"]

[lab05]
small_model = "openai/gpt-oss-20b"
large_model = "openai/gpt-oss-20b"

[lab09]
batch_completion_window = "24h"

[lab04]
{lab04_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("model", "openai/gpt-oss-20b")
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def _retry_policy_answer():
    return (
        "max_retry_attempts: 4\n"
        "max_total_attempts: 5\n"
        "base_delay_seconds: 1\n"
        "last_nominal_delay_seconds: 8\n"
        "retryable_http_status: 429\n"
    )


def _invocation_single_answer():
    return "capital_of_japan: Tokyo\nboiling_point_celsius: 100\nhamlet_author: William Shakespeare\n"


def test_dry_run_never_calls_api_and_writes_plan(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, output_limit_values=[64, 256, 1024], output_limit_repetitions=2, max_api_calls=10)
    parser = build_parser()
    args = parser.parse_args(["lab04", "all", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab04_output_control").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 10  # header + 10 planned calls (default Grand Test)


def test_without_live_flag_no_api_call_is_made(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("must never construct a real client without --live")

    config_path = _write_config(tmp_path, live=False, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab04", "output_limit_sweep", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False


def test_disabled_lab04_makes_no_api_call(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, enabled=False, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("disabled lab04 must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab04", "all", "--live", "--config", str(config_path)])
    args.func(args)  # must not raise


def test_output_limit_sweep_writes_measurements_and_detects_truncation(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, output_limit_values=[16, 1024], output_limit_repetitions=1, max_api_calls=10)
    responses = [
        make_response(content="max_retry_attempts: 4", finish_reason="length", prompt_tokens=88, completion_tokens=16, reasoning_tokens=16),
        make_response(content=_retry_policy_answer(), finish_reason="stop", prompt_tokens=88, completion_tokens=40, reasoning_tokens=10),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab04", "output_limit_sweep", "--live", "--experiment-run-id", "ol-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab04_output_control" / "ol-test"
    assert (run_dir / "output_limit_measurements.json").exists()
    assert not (run_dir / "invocation_measurements.json").exists() or json.loads((run_dir / "invocation_measurements.json").read_text(encoding="utf-8")) == []

    with open(run_dir / "output_limit_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 2
    by_tokens = {r["max_completion_tokens"]: r for r in records}
    assert by_tokens[16]["output_truncated"] is True
    assert by_tokens[16]["overall_answer_accepted"] is False
    assert by_tokens[1024]["output_truncated"] is False
    assert by_tokens[1024]["overall_answer_accepted"] is True


def test_invocation_comparison_runs_exactly_four_calls(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    responses = [
        make_response(content=_invocation_single_answer()),
        make_response(content="Tokyo"),
        make_response(content="100"),
        make_response(content="William Shakespeare"),
    ]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab04", "invocation_comparison", "--live", "--experiment-run-id", "inv-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 4
    run_dir = tmp_path / "results" / "lab04_output_control" / "inv-test"
    with open(run_dir / "invocation_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 4
    assert all(r["overall_answer_accepted"] for r in records)
    with open(run_dir / "invocation_comparison.csv", encoding="utf-8") as f:
        rows = f.read().strip().splitlines()
    assert len(rows) - 1 == 2  # header + single/multiple rows


def test_all_mode_combines_both_experiments_and_renders_nine_charts(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    responses = [
        make_response(content=_retry_policy_answer(), prompt_tokens=88, completion_tokens=40, reasoning_tokens=10),
        make_response(content=_invocation_single_answer()),
        make_response(content="Tokyo"),
        make_response(content="100"),
        make_response(content="William Shakespeare"),
    ]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab04", "all", "--live", "--experiment-run-id", "all-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab04_output_control" / "all-test"
    assert (run_dir / "experiment_summary.json").exists()
    assert (run_dir / "analysis_report.md").exists()

    charts_dir = run_dir / "charts"
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9
    for png in png_files:
        assert png.stat().st_size > 0


def test_cli_override_exceeding_budget_is_refused_with_zero_api_calls(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("budget-exceeding override must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, output_limit_values=[64, 256, 1024], output_limit_repetitions=2, max_api_calls=10)
    parser = build_parser()
    args = parser.parse_args(["lab04", "all", "--live", "--repetitions", "20", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    assert not (tmp_path / "results" / "lab04_output_control").exists() or not any(
        (tmp_path / "results" / "lab04_output_control").glob("*/output_limit_measurements.json")
    )


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    responses = [
        make_response(content=_retry_policy_answer()),
        make_response(content=_invocation_single_answer()),
        make_response(content="Tokyo"),
        make_response(content="100"),
        make_response(content="William Shakespeare"),
    ]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab04", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("resuming a used run_id must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    args2 = parser.parse_args(["lab04", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)

    assert client_constructed["called"] is False


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    responses = [make_response(content=_retry_policy_answer())]
    _patch_groq(monkeypatch, responses)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab04_output_control.charts.generate_output_control_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab04", "output_limit_sweep", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab04_output_control" / "chart-fail"
    assert (run_dir / "output_limit_measurements.json").exists()
    assert (run_dir / "experiment_summary.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_runs_without_live_or_api_key(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, output_limit_values=[64], output_limit_repetitions=1, max_api_calls=10)
    responses = [make_response(content=_retry_policy_answer())]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab04", "output_limit_sweep", "--live", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("backfill must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backfill_args = parser.parse_args(["lab04", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert client_constructed["called"] is False
