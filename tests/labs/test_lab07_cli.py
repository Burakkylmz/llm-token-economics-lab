"""Offline end-to-end tests for the `lab07` CLI. Lab07 never constructs a
Groq client or needs an API key in any mode, so these tests never
monkeypatch Groq at all -- that absence is itself part of what's being
verified (no accidental import/construction of a real client anywhere in
the command path).
"""

from __future__ import annotations

import json

import pytest

from lab_core.cli import build_parser


def _fmt_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _toml(tmp_path, **lab07_overrides) -> str:
    lab07_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab07_overrides.items())
    return f"""
[groq]
default_model = "openai/gpt-oss-20b"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = false
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

[lab07]
{lab07_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def test_default_grand_test_runs_twelve_simulation_runs_and_renders_nine_charts(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--experiment-run-id", "default-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "default-test"
    assert (run_dir / "raw_measurements.json").exists()
    assert (run_dir / "attempt_timeline.csv").exists()
    assert (run_dir / "strategy_summary.csv").exists()
    assert (run_dir / "scenario_comparison.csv").exists()
    assert (run_dir / "analysis_report.md").exists()

    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["real_api_calls"] == 0
    assert summary["real_token_usage"] is None
    strategy_summaries = summary["strategy_summaries"]
    scenarios = {s["scenario_id"] for s in strategy_summaries}
    assert len(scenarios) == 3
    assert len(strategy_summaries) == 12

    charts_dir = run_dir / "charts"
    png_files = list(charts_dir.glob("*.png"))
    assert len(png_files) == 9
    for png in png_files:
        assert png.stat().st_size > 0


def test_pilot_profile_runs_four_simulation_runs(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "pilot", "--experiment-run-id", "pilot-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "pilot-test"
    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert len(summary["strategy_summaries"]) == 4


def test_extended_profile_runs_twenty_simulation_runs(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "extended", "--experiment-run-id", "extended-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "extended-test"
    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert len(summary["strategy_summaries"]) == 20
    scenarios = {s["scenario_id"] for s in summary["strategy_summaries"]}
    assert scenarios == {"transient_failures", "server_recovery_window", "shared_contention", "retry_after_variations", "quota_pressure"}


def test_single_strategy_backward_compatible_command(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "exponential_backoff_with_jitter", "--experiment-run-id", "single-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "single-test"
    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    strategies = {s["strategy_id"] for s in summary["strategy_summaries"]}
    assert strategies == {"exponential_backoff_with_jitter"}


def test_legacy_fail_count_flag_forces_single_scenario(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "no_backoff", "--fail-count", "4", "--experiment-run-id", "legacy-test", "--config", str(config_path)])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "legacy-test"
    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert len(summary["strategy_summaries"]) == 1
    assert summary["strategy_summaries"][0]["scenario_id"] == "transient_failures"
    assert summary["strategy_summaries"][0]["total_429_responses"] == 4


def test_legacy_retry_after_seconds_flag_injects_header(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args([
        "lab07", "rate_limit_aware", "--fail-count", "1", "--retry-after-seconds", "9",
        "--experiment-run-id", "legacy-header-test", "--config", str(config_path),
    ])
    args.func(args)

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "legacy-header-test"
    with open(run_dir / "attempt_timeline.csv", encoding="utf-8") as f:
        content = f.read()
    assert "retry_after_header" in content


def test_live_flag_is_ignored_and_prints_explicit_warning(tmp_path, capsys):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "pilot", "--live", "--experiment-run-id", "live-ignored", "--config", str(config_path)])
    args.func(args)

    out = capsys.readouterr().out
    assert "LAB07 IS SIMULATION ONLY" in out
    assert "No real API calls will be made" in out
    assert "--live was passed but is IGNORED" in out


def test_disabled_lab07_does_not_execute(tmp_path):
    config_path = _write_config(tmp_path, enabled=False)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--experiment-run-id", "disabled-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not (tmp_path / "results" / "lab07_rate_limiting" / "disabled-test").exists()


def test_duplicate_experiment_run_id_is_refused(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "pilot", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)
    args2 = parser.parse_args(["lab07", "all", "--profile", "pilot", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)  # must not raise, must refuse silently-safely

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "dup-run"
    with open(run_dir / "raw_measurements.json", encoding="utf-8") as f:
        records = json.load(f)
    assert len(records) == 4  # pilot profile's original run, never duplicated/overwritten


def test_client_count_exceeding_max_clients_is_rejected_at_config_load(tmp_path):
    # Validated fail-closed at config-load time (lab_core.config._build_lab07)
    # -- a contention/quota client count above max_clients can never even
    # produce a loadable config, so the CLI's own runtime guard for this is
    # purely defensive and never reachable from a valid config file.
    from lab_core.config import ConfigError, load_config

    config_path = _write_config(tmp_path, max_clients=2, contention_client_count=8)
    with pytest.raises(ConfigError):
        load_config(config_path)


def test_chart_generation_failure_does_not_lose_measurements(tmp_path, monkeypatch, capsys):
    config_path = _write_config(tmp_path)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab07_rate_limiting.charts.generate_retry_charts", _boom)

    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "pilot", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = tmp_path / "results" / "lab07_rate_limiting" / "chart-fail"
    assert (run_dir / "raw_measurements.json").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_command_requires_no_api_key(tmp_path, monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab07", "all", "--profile", "pilot", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    backfill_args = parser.parse_args(["lab07", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)  # must not raise


def test_seed_cli_override_changes_jitter_trajectory(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()

    args_a = parser.parse_args(["lab07", "exponential_backoff_with_jitter", "--fail-count", "4", "--seed", "1", "--experiment-run-id", "seed-a", "--config", str(config_path)])
    args_a.func(args_a)
    args_b = parser.parse_args(["lab07", "exponential_backoff_with_jitter", "--fail-count", "4", "--seed", "2", "--experiment-run-id", "seed-b", "--config", str(config_path)])
    args_b.func(args_b)

    run_dir_a = tmp_path / "results" / "lab07_rate_limiting" / "seed-a"
    run_dir_b = tmp_path / "results" / "lab07_rate_limiting" / "seed-b"
    content_a = (run_dir_a / "attempt_timeline.csv").read_text(encoding="utf-8")
    content_b = (run_dir_b / "attempt_timeline.csv").read_text(encoding="utf-8")
    assert content_a != content_b
