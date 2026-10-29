from __future__ import annotations

from lab_core.cli import build_parser

EXPECTED_LAB_IDS = [f"lab0{i}" for i in range(1, 10)] + ["lab10"]


def test_all_ten_labs_are_registered():
    parser = build_parser()
    subparsers_action = next(action for action in parser._actions if action.dest == "lab")
    assert sorted(subparsers_action.choices.keys()) == sorted(EXPECTED_LAB_IDS)


def test_lab01_without_live_prints_guard_message_and_writes_nothing(capsys, tmp_path):
    parser = build_parser()
    args = parser.parse_args(["lab01", "short_prompt"])
    args.func(args)
    captured = capsys.readouterr()
    assert "requires --live" in captured.out
    assert not (tmp_path / "lab01_token_anatomy.json").exists()


def test_lab07_runs_fully_simulated_without_live_using_a_temp_results_dir(capsys, tmp_path):
    results_dir = tmp_path / "results"
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
[groq]
default_model = "llama-3.1-8b-instant"
request_timeout_seconds = 30.0

[execution]
live_api_enabled = false
cost_warning_message = "warning"

[experiment]
default_repetitions = 2
results_dir = "{results_dir.as_posix()}"

[retry]
max_attempts = 3
base_delay_seconds = 0.01
max_delay_seconds = 0.02
jitter = false

[concurrency]
max_concurrent_requests = 2

[lab03]
caching_capable_models = ["openai/gpt-oss-20b"]

[lab05]
small_model = "llama-3.1-8b-instant"
large_model = "llama-3.3-70b-versatile"

[lab09]
batch_completion_window = "24h"
""",
        encoding="utf-8",
    )

    parser = build_parser()
    args = parser.parse_args(
        ["lab07", "no_backoff", "--fail-count", "1", "--experiment-run-id", "smoke-test", "--config", str(config_path)]
    )
    args.func(args)

    captured = capsys.readouterr()
    assert "LAB07 IS SIMULATION ONLY" in captured.out
    assert "No real API calls will be made" in captured.out
    assert (results_dir / "lab07_rate_limiting" / "smoke-test" / "raw_measurements.json").exists()
