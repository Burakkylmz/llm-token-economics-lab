"""Offline, mocked end-to-end tests for the `lab05` CLI: dry-run safety
(including offline routing-decision preview), live execution (fake Groq
client, no real network), request-budget refusal, duplicate-run-id
refusal, automatic chart generation, and the `lab05 backfill` command. No
real Groq API calls anywhere.
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


def _toml(tmp_path, live=True, **lab05_overrides) -> str:
    lab05_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab05_overrides.items())
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

[pricing.models."openai/gpt-oss-120b"]
input_per_1m_usd = 0.15
output_per_1m_usd = 0.60
cached_input_per_1m_usd = 0.075
price_verified_date = "2026-10-21"

[lab03]
caching_capable_models = ["openai/gpt-oss-20b"]

[lab09]
batch_completion_window = "24h"

[lab05]
{lab05_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("small_model", "openai/gpt-oss-20b")
    kwargs.setdefault("large_model", "openai/gpt-oss-120b")
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def test_dry_run_never_calls_api_and_shows_routing_decisions(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, default_profile="default", max_api_calls=18)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab05_model_routing").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 12  # header + default Grand Test's 12 calls


def test_pilot_profile_dry_run_plans_eight_calls(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=18)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--dry-run", "--profile", "pilot", "--config", str(config_path)])
    args.func(args)

    plan_files = list((tmp_path / "results" / "lab05_model_routing").glob("*/experiment_plan.csv"))
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 8


def test_extended_profile_dry_run_plans_eighteen_calls(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=18)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--dry-run", "--profile", "extended", "--config", str(config_path)])
    args.func(args)

    plan_files = list((tmp_path / "results" / "lab05_model_routing").glob("*/experiment_plan.csv"))
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 18


def test_dry_run_prints_offline_routing_decisions(monkeypatch, tmp_path, capsys):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=18)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--dry-run", "--profile", "pilot", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "Routing Decisions (computed offline" in out
    assert "task_009" in out or "task_010" in out  # pilot includes task_010
    assert "Estimated Maximum Cost" in out


def test_without_live_flag_no_api_call_is_made(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("must never construct a real client without --live")

    config_path = _write_config(tmp_path, live=False, max_api_calls=18)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab05", "fixed_model_baseline", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False


def test_fixed_model_baseline_uses_only_small_model(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(6)]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab05", "fixed_model_baseline", "--experiment-run-id", "fixed-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 6
    assert all(c["model"] == "openai/gpt-oss-20b" for c in fake.completions.calls)


def test_always_large_uses_only_large_model(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(6)]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab05", "always_large", "--experiment-run-id", "large-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 6
    assert all(c["model"] == "openai/gpt-oss-120b" for c in fake.completions.calls)


def test_default_grand_test_runs_exactly_twelve_calls_and_renders_nine_charts(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(12)]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--experiment-run-id", "all-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 12
    run_dir = tmp_path / "results" / "lab05_model_routing" / "all-test"
    assert (run_dir / "routing_measurements.json").exists()
    assert (run_dir / "strategy_summary.csv").exists()
    assert (run_dir / "task_comparison.csv").exists()
    assert (run_dir / "analysis_report.md").exists()

    charts_dir = run_dir / "charts"
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9
    for png in png_files:
        assert png.stat().st_size > 0


def test_routing_reason_and_model_tier_persisted(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(12)]
    _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--experiment-run-id", "reason-test", "--live", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab05_model_routing" / "reason-test"
    with open(run_dir / "routing_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert all(r["routing_reason"] for r in records)
    router_records = [r for r in records if r["strategy"] == "rule_based_routing"]
    assert any(r["model_tier"] == "large" for r in router_records)  # task_009/task_011 route large


def test_cli_profile_override_exceeding_budget_is_refused(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("budget-exceeding profile must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=10)  # below extended's 18
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--live", "--profile", "extended", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    assert not (tmp_path / "results" / "lab05_model_routing").exists() or not any(
        (tmp_path / "results" / "lab05_model_routing").glob("*/routing_measurements.json")
    )


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(12)]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("resuming a used run_id must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    args2 = parser.parse_args(["lab05", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)

    assert client_constructed["called"] is False


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(12)]
    _patch_groq(monkeypatch, responses)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab05_model_routing.charts.generate_routing_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab05_model_routing" / "chart-fail"
    assert (run_dir / "routing_measurements.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_runs_without_live_or_api_key(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, max_api_calls=18)
    responses = [make_response(content="Rome") for _ in range(12)]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab05", "all", "--live", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("backfill must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backfill_args = parser.parse_args(["lab05", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert client_constructed["called"] is False
