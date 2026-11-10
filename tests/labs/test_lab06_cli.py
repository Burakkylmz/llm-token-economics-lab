"""Offline, mocked end-to-end tests for the `lab06` CLI: dry-run safety,
live execution (fake Groq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, and the
`lab06 backfill` command. No real Groq API calls anywhere.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from lab_core.cli import build_parser


def _choice(content=None, finish_reason=None, role=None):
    delta = SimpleNamespace(content=content, reasoning=None, role=role)
    return SimpleNamespace(delta=delta, finish_reason=finish_reason)


def _stream_chunks():
    return [
        SimpleNamespace(choices=[_choice(role="assistant")], usage=None, x_groq=None, system_fingerprint=None),
        SimpleNamespace(choices=[_choice(content="Incident Summary: ok Possible Causes: ok")], usage=None, x_groq=None, system_fingerprint=None),
        SimpleNamespace(choices=[_choice(content="", finish_reason="stop")], usage=None, x_groq=None, system_fingerprint=None),
    ]


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
    return str(v)


def _toml(tmp_path, live=True, **lab06_overrides) -> str:
    lab06_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab06_overrides.items())
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

[lab06]
{lab06_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("model", "openai/gpt-oss-20b")
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

    config_path = _write_config(tmp_path, paired_repetitions=5, max_api_calls=16)
    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--dry-run", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    plan_files = list((tmp_path / "results" / "lab06_streaming").glob("*/experiment_plan.csv"))
    assert len(plan_files) == 1
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 10  # header + default Grand Test's 10 calls


def test_pilot_profile_dry_run_plans_four_calls(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=16)
    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--dry-run", "--profile", "pilot", "--config", str(config_path)])
    args.func(args)

    plan_files = list((tmp_path / "results" / "lab06_streaming").glob("*/experiment_plan.csv"))
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 4


def test_extended_profile_dry_run_plans_sixteen_calls(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=16)
    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--dry-run", "--profile", "extended", "--config", str(config_path)])
    args.func(args)

    plan_files = list((tmp_path / "results" / "lab06_streaming").glob("*/experiment_plan.csv"))
    with open(plan_files[0], encoding="utf-8") as f:
        assert sum(1 for _ in f) - 1 == 16


def test_without_live_flag_no_api_call_is_made(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("must never construct a real client without --live")

    config_path = _write_config(tmp_path, live=False, max_api_calls=16)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab06", "non_streaming", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False


def test_disabled_lab06_makes_no_api_call(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, enabled=False, max_api_calls=16)

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("disabled lab06 must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--live", "--config", str(config_path)])
    args.func(args)  # must not raise


def test_non_streaming_single_condition_runs_only_non_streaming(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, paired_repetitions=2, max_api_calls=16)
    responses = [make_response(content="Incident Summary: ok Possible Causes: ok") for _ in range(2)]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab06", "non_streaming", "--experiment-run-id", "ns-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 2
    assert all(c.get("stream") in (False, None) for c in fake.completions.calls)
    run_dir = tmp_path / "results" / "lab06_streaming" / "ns-test"
    with open(run_dir / "stream_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert all(r["condition"] == "non_streaming" for r in records)


def test_streaming_single_condition_runs_only_streaming(monkeypatch, tmp_path):
    config_path = _write_config(tmp_path, paired_repetitions=2, max_api_calls=16)
    responses = [_stream_chunks(), _stream_chunks()]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab06", "streaming", "--experiment-run-id", "s-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 2
    run_dir = tmp_path / "results" / "lab06_streaming" / "s-test"
    with open(run_dir / "stream_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert all(r["condition"] == "streaming" for r in records)
    assert (run_dir / "stream_events.csv").exists()


def test_default_grand_test_runs_exactly_ten_calls_and_renders_nine_charts(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, paired_repetitions=1, max_api_calls=16)
    # pair 1 default order: non_streaming, streaming.
    responses = [make_response(content="Incident Summary: ok"), _stream_chunks()]
    fake = _patch_groq(monkeypatch, responses)

    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--experiment-run-id", "all-test", "--live", "--config", str(config_path)])
    args.func(args)

    assert len(fake.completions.calls) == 2
    run_dir = tmp_path / "results" / "lab06_streaming" / "all-test"
    assert (run_dir / "stream_measurements.json").exists()
    assert (run_dir / "paired_comparison.csv").exists()
    assert (run_dir / "condition_summary.csv").exists()
    assert (run_dir / "analysis_report.md").exists()

    charts_dir = run_dir / "charts"
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9
    for png in png_files:
        assert png.stat().st_size > 0


def test_cli_profile_override_exceeding_budget_is_refused(monkeypatch, tmp_path):
    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("budget-exceeding profile must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, max_api_calls=10)  # below extended's 16
    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--live", "--profile", "extended", "--config", str(config_path)])
    args.func(args)

    assert client_constructed["called"] is False
    assert not (tmp_path / "results" / "lab06_streaming").exists() or not any(
        (tmp_path / "results" / "lab06_streaming").glob("*/stream_measurements.json")
    )


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, paired_repetitions=1, max_api_calls=16)
    responses = [make_response(content="Incident Summary: ok"), _stream_chunks()]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab06", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("resuming a used run_id must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    args2 = parser.parse_args(["lab06", "all", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)

    assert client_constructed["called"] is False


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, make_response, capsys):
    config_path = _write_config(tmp_path, paired_repetitions=1, max_api_calls=16)
    responses = [make_response(content="Incident Summary: ok")]
    _patch_groq(monkeypatch, responses)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab06_streaming.charts.generate_streaming_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab06", "non_streaming", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab06_streaming" / "chart-fail"
    assert (run_dir / "stream_measurements.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_runs_without_live_or_api_key(monkeypatch, tmp_path, make_response):
    config_path = _write_config(tmp_path, paired_repetitions=1, max_api_calls=16)
    responses = [make_response(content="Incident Summary: ok")]
    _patch_groq(monkeypatch, responses)
    parser = build_parser()
    args = parser.parse_args(["lab06", "non_streaming", "--live", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    client_constructed = {"called": False}

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            client_constructed["called"] = True
            raise AssertionError("backfill must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    backfill_args = parser.parse_args(["lab06", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert client_constructed["called"] is False
