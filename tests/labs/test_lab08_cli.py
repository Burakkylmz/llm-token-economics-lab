"""Offline, mocked end-to-end tests for the `lab08` CLI: dry-run safety,
live execution (fake AsyncGroq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, and the
`lab08 backfill` command. Lab08 constructs its OWN AsyncGroq client
directly in `labs.lab08_concurrency.cli` (never the shared
`lab_core.lab_cli_support` client), so that module path is what gets
monkeypatched here.
"""

from __future__ import annotations

import asyncio
import csv
import json
from types import SimpleNamespace

import pytest

from lab_core.cli import build_parser


def _usage():
    return SimpleNamespace(
        prompt_tokens=10, completion_tokens=5, total_tokens=15, total_time=0.01,
        prompt_tokens_details=None, completion_tokens_details=None,
    )


def _response(content="2"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=_usage(), system_fingerprint=None)


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []
        self.construct_kwargs: dict | None = None

    async def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        await asyncio.sleep(0.001)
        return _response()


def _make_fake_async_groq_class():
    completions = _FakeCompletions()
    construct_calls: list[dict] = []

    class _FakeAsyncGroq:
        def __init__(self, **kwargs):
            construct_calls.append(dict(kwargs))
            self.chat = SimpleNamespace(completions=completions)

    _FakeAsyncGroq.completions = completions
    _FakeAsyncGroq.construct_calls = construct_calls
    return _FakeAsyncGroq


def _patch_groq(monkeypatch):
    fake_class = _make_fake_async_groq_class()
    monkeypatch.setattr("labs.lab08_concurrency.cli.AsyncGroq", fake_class)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return fake_class


def _fmt_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_toml_value(x) for x in v) + "]"
    return str(v)


def _toml(tmp_path, live=True, **lab08_overrides) -> str:
    lab08_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab08_overrides.items())
    return f"""
[groq]
default_model = "openai/gpt-oss-20b"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = {"true" if live else "false"}
cost_warning_message = "warning"

[experiment]
default_repetitions = 1
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

[lab08]
{lab08_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("model", "openai/gpt-oss-20b")
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def _run_dir(tmp_path, run_id):
    return tmp_path / "results" / "lab08_concurrency" / run_id


def test_dry_run_never_constructs_a_client_and_writes_plan(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("labs.lab08_concurrency.cli.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path, concurrency_levels=[1, 5])
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--dry-run", "--experiment-run-id", "dry-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = _run_dir(tmp_path, "dry-test")
    with open(run_dir / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 16  # default profile: 8 tasks x 2 levels
    assert not (run_dir / "raw_measurements.json").exists()


def test_dry_run_pilot_profile_plans_eight_calls(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--dry-run", "--profile", "pilot", "--experiment-run-id", "pilot-dry", "--config", str(config_path)])
    args.func(args)

    with open(_run_dir(tmp_path, "pilot-dry") / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8


def test_dry_run_extended_profile_plans_thirtytwo_calls(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--dry-run", "--profile", "extended", "--experiment-run-id", "ext-dry", "--config", str(config_path)])
    args.func(args)

    with open(_run_dir(tmp_path, "ext-dry") / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 32


def test_live_default_run_produces_all_result_files_and_charts(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, concurrency_levels=[1, 5], max_api_calls=16)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--live", "--experiment-run-id", "live-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_class.completions.calls) == 16

    run_dir = _run_dir(tmp_path, "live-test")
    assert (run_dir / "raw_measurements.json").exists()
    assert (run_dir / "request_timeline.csv").exists()
    assert (run_dir / "condition_summary.csv").exists()
    assert (run_dir / "concurrency_comparison.csv").exists()
    assert (run_dir / "analysis_report.md").exists()

    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["planned_calls"] == 16
    assert summary["successful_calls"] == 16
    assert len(summary["conditions"]) == 2

    charts_dir = run_dir / "charts"
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9


def test_sdk_max_retries_applied_only_to_this_labs_own_client(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, sdk_max_retries=3, concurrency_levels=[1])
    parser = build_parser()
    args = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "retries-test", "--config", str(config_path)])
    args.func(args)

    assert fake_class.construct_calls[-1]["max_retries"] == 3


def test_sequential_command_runs_only_concurrency_one(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, concurrency_levels=[1, 5])
    parser = build_parser()
    args = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "seq-test", "--config", str(config_path)])
    args.func(args)

    with open(_run_dir(tmp_path, "seq-test") / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    levels = {c["max_concurrent"] for c in summary["conditions"]}
    assert levels == {1}
    assert summary["planned_calls"] == 8  # full 8-task workload at concurrency=1


def test_concurrent_command_respects_max_concurrent_override(monkeypatch, tmp_path):
    _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, concurrency_levels=[1, 5])
    parser = build_parser()
    args = parser.parse_args(["lab08", "concurrent", "--max-concurrent", "3", "--live", "--experiment-run-id", "conc-test", "--config", str(config_path)])
    args.func(args)

    with open(_run_dir(tmp_path, "conc-test") / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    levels = {c["max_concurrent"] for c in summary["conditions"]}
    assert levels == {3}


def test_zero_max_concurrent_override_is_rejected(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab08", "concurrent", "--max-concurrent", "0", "--dry-run", "--experiment-run-id", "zero-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not _run_dir(tmp_path, "zero-test").exists()


def test_concurrency_exceeding_max_allowed_concurrency_is_rejected_at_config_load(tmp_path):
    # Validated fail-closed at config-load time (lab_core.config._build_lab08)
    # -- same precedent as Lab07Config -- so a default-profile level above
    # the ceiling can never even produce a loadable config.
    from lab_core.config import ConfigError, load_config

    config_path = _write_config(tmp_path, concurrency_levels=[1, 5], max_allowed_concurrency=4)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_max_api_calls_budget_refusal(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("must never construct a client when the budget is refused")

    monkeypatch.setattr("labs.lab08_concurrency.cli.AsyncGroq", _ExplodingGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    config_path = _write_config(tmp_path, concurrency_levels=[1, 5], max_api_calls=4)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--live", "--experiment-run-id", "budget-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not _run_dir(tmp_path, "budget-test").exists()


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, concurrency_levels=[1])
    parser = build_parser()

    args = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)
    first_call_count = len(fake_class.completions.calls)

    args2 = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)  # must not raise, must refuse without making new calls

    assert len(fake_class.completions.calls) == first_call_count

    with open(_run_dir(tmp_path, "dup-run") / "raw_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 8  # original run only, never duplicated


def test_disabled_lab08_does_not_execute(tmp_path):
    config_path = _write_config(tmp_path, enabled=False)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--live", "--experiment-run-id", "disabled-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not _run_dir(tmp_path, "disabled-test").exists()


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, capsys):
    _patch_groq(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab08_concurrency.charts.generate_concurrency_charts", _boom)

    config_path = _write_config(tmp_path, concurrency_levels=[1])
    parser = build_parser()
    args = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = _run_dir(tmp_path, "chart-fail")
    assert (run_dir / "raw_measurements.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_requires_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab08", "backfill", "--config", str(config_path)])
    args.func(args)  # must not raise even with no legacy file present and no API key


def test_live_requires_live_flag_otherwise_refuses(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab08", "all", "--experiment-run-id", "no-live-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not _run_dir(tmp_path, "no-live-test").exists()


def test_one_failed_request_is_recorded_without_losing_others(monkeypatch, tmp_path):
    completions = _FakeCompletions()
    call_count = {"n": 0}

    async def _create(**kwargs):
        call_count["n"] += 1
        completions.calls.append(dict(kwargs))
        if call_count["n"] == 3:
            raise ConnectionError("simulated failure")
        return _response()

    completions.create = _create

    class _FakeAsyncGroq:
        def __init__(self, **kwargs):
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr("labs.lab08_concurrency.cli.AsyncGroq", _FakeAsyncGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    config_path = _write_config(tmp_path, concurrency_levels=[1])
    parser = build_parser()
    args = parser.parse_args(["lab08", "sequential", "--live", "--experiment-run-id", "partial-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    with open(_run_dir(tmp_path, "partial-fail") / "raw_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 8
    statuses = [r["request_status"] for r in records]
    assert statuses.count("SUCCESS") == 7
    assert statuses.count("UNKNOWN_ERROR") == 1
