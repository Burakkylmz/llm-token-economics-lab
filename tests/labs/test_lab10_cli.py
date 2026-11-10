"""Offline, mocked end-to-end tests for the `lab10` CLI: dry-run safety,
live execution (fake Groq client, no real network), request-budget
refusal, duplicate-run-id refusal, automatic chart generation, `evaluate`
reusing an existing summary without regenerating it, and `lab10 backfill`.
Lab10 uses the shared SYNC `Groq` client from `lab_core.lab_cli_support`
(never a dedicated client), so that module's `Groq` is what gets patched.
"""

from __future__ import annotations

import csv
import json
from types import SimpleNamespace

from lab_core.cli import build_parser


def _usage(pt=200, ct=30, rt=5):
    return SimpleNamespace(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct, total_time=0.1,
                            prompt_tokens_details=None, completion_tokens_details=SimpleNamespace(reasoning_tokens=rt))


def _resp(content, finish_reason="stop", **kwargs):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=_usage(**kwargs), system_fingerprint=None)


VALID_SUMMARY_JSON = json.dumps({
    "overview": "o", "facts": [{"name": "order_id", "value": "ORD-74291", "status": "confirmed"}],
    "decisions": [], "open_issues": [], "timeline": [],
})


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(dict(kwargs))
        user_content = kwargs["messages"][-1]["content"]
        if "Question:\n" in user_content:
            return _resp(json.dumps({"answer": "ORD-74291"}), pt=150, ct=15)
        return _resp(VALID_SUMMARY_JSON, pt=1800, ct=300)


def _make_fake_groq_class():
    completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self, api_key=None, timeout=None, max_retries=None):
            self.api_key = api_key
            self.chat = SimpleNamespace(completions=completions)

    _FakeClient.completions = completions
    return _FakeClient


def _patch_groq(monkeypatch):
    fake_class = _make_fake_groq_class()
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", fake_class)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return fake_class


def _fmt_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    return str(v)


def _toml(tmp_path, live=True, **lab10_overrides) -> str:
    lab10_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab10_overrides.items())
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

[lab10]
{lab10_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def _run_dir(tmp_path, run_id):
    return tmp_path / "results" / "lab10_conversation_summarization" / run_id


def test_dry_run_never_constructs_a_client_and_writes_plan(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)

    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--dry-run", "--profile", "default", "--experiment-run-id", "dry-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = _run_dir(tmp_path, "dry-test")
    with open(run_dir / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 14
    assert not (run_dir / "summary_measurements.csv").exists()


def test_dry_run_pilot_profile_plans_eight_calls(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--dry-run", "--profile", "pilot", "--experiment-run-id", "pilot-dry", "--config", str(config_path)])
    args.func(args)
    with open(_run_dir(tmp_path, "pilot-dry") / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8


def test_dry_run_extended_profile_plans_twentyeight_calls(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--dry-run", "--profile", "extended", "--experiment-run-id", "ext-dry", "--config", str(config_path)])
    args.func(args)
    with open(_run_dir(tmp_path, "ext-dry") / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 28


def test_live_default_run_produces_all_result_files_and_charts(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "default", "--experiment-run-id", "live-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_class.completions.calls) == 14

    run_dir = _run_dir(tmp_path, "live-test")
    assert (run_dir / "summary_measurements.csv").exists()
    assert (run_dir / "followup_measurements.csv").exists()
    assert (run_dir / "generated_summaries.json").exists()
    assert (run_dir / "request_response_inspection.json").exists()
    assert (run_dir / "context_comparison.csv").exists()
    assert (run_dir / "token_comparison.csv").exists()
    assert (run_dir / "cost_comparison.csv").exists()
    assert (run_dir / "quality_evaluation.json").exists()
    assert (run_dir / "break_even_analysis.csv").exists()
    assert (run_dir / "analysis_report.md").exists()

    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["planned_calls"] == 14
    assert len(summary["conditions"]) == 3

    charts_dir = run_dir / "charts"
    assert len(list(charts_dir.glob("*.png"))) == 6


def test_summarize_command_makes_no_followup_calls(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "summarize", "--live", "--profile", "pilot", "--experiment-run-id", "summarize-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_class.completions.calls) == 2
    for call in fake_class.completions.calls:
        assert "Question:\n" not in call["messages"][-1]["content"]


def test_evaluate_reuses_existing_summary_without_regenerating(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()

    args = parser.parse_args(["lab10", "summarize", "--live", "--profile", "pilot", "--experiment-run-id", "source-run", "--config", str(config_path)])
    args.func(args)
    calls_after_summarize = len(fake_class.completions.calls)
    assert calls_after_summarize == 2

    args2 = parser.parse_args(["lab10", "evaluate", "--live", "--profile", "pilot", "--experiment-run-id", "source-run", "--config", str(config_path)])
    args2.func(args2)

    # evaluate must add exactly the follow-up calls (pilot: 2 questions x 3
    # conditions = 6), never another summary-generation call.
    assert len(fake_class.completions.calls) == calls_after_summarize + 6
    summary_gen_calls = [c for c in fake_class.completions.calls if "Question:\n" not in c["messages"][-1]["content"]]
    assert len(summary_gen_calls) == 2  # still only the original 2, never regenerated


def test_evaluate_without_existing_summary_fails_clearly(monkeypatch, tmp_path, capsys):
    _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "evaluate", "--live", "--profile", "pilot", "--experiment-run-id", "no-such-run", "--config", str(config_path)])
    args.func(args)  # must not raise
    out = capsys.readouterr().out
    assert "no generated summaries" in out.lower()
    assert not _run_dir(tmp_path, "no-such-run-evaluate").exists()


def test_evaluate_without_run_id_fails_clearly(monkeypatch, tmp_path, capsys):
    _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "evaluate", "--live", "--profile", "pilot", "--config", str(config_path)])
    args.func(args)  # must not raise
    assert "requires --experiment-run-id" in capsys.readouterr().out


def test_max_api_calls_budget_refusal(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("must never construct a client when the budget is refused")

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    config_path = _write_config(tmp_path, max_api_calls=4)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "default", "--experiment-run-id", "budget-test", "--config", str(config_path)])
    args.func(args)  # must not raise

    assert not _run_dir(tmp_path, "budget-test").exists()


def test_duplicate_experiment_run_id_is_refused(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()

    args = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args.func(args)
    first_call_count = len(fake_class.completions.calls)

    args2 = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "dup-run", "--config", str(config_path)])
    args2.func(args2)  # must not raise, must refuse without making new calls

    assert len(fake_class.completions.calls) == first_call_count

    with open(_run_dir(tmp_path, "dup-run") / "followup_measurements.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6  # original pilot run only (2 questions x 3 conditions), never duplicated


def test_disabled_lab10_does_not_execute(tmp_path):
    config_path = _write_config(tmp_path, enabled=False)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "disabled-test", "--config", str(config_path)])
    args.func(args)  # must not raise
    assert not _run_dir(tmp_path, "disabled-test").exists()


def test_chart_generation_failure_does_not_lose_measurements(monkeypatch, tmp_path, capsys):
    _patch_groq(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("simulated chart failure")

    monkeypatch.setattr("labs.lab10_conversation_summarization.charts.generate_summarization_charts", _boom)

    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "chart-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    run_dir = _run_dir(tmp_path, "chart-fail")
    assert (run_dir / "followup_measurements.csv").exists()
    assert "Chart generation failed" in capsys.readouterr().out


def test_backfill_requires_no_api_key(monkeypatch, tmp_path):
    fake_class = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "backfill-src", "--config", str(config_path)])
    args.func(args)

    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setattr("lab_core.lab_cli_support.Groq", lambda api_key=None: (_ for _ in ()).throw(AssertionError("backfill must never construct a client")))

    backfill_args = parser.parse_args(["lab10", "backfill", "--config", str(config_path)])
    backfill_args.func(backfill_args)  # must not raise

    run_dir = _run_dir(tmp_path, "backfill-src")
    assert len(list((run_dir / "charts").glob("*.png"))) == 6


def test_live_requires_live_flag_otherwise_refuses(tmp_path):
    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--profile", "pilot", "--experiment-run-id", "no-live-test", "--config", str(config_path)])
    args.func(args)  # must not raise
    assert not _run_dir(tmp_path, "no-live-test").exists()


def test_one_failed_followup_is_recorded_without_losing_others(monkeypatch, tmp_path):
    completions = _FakeCompletions()
    call_count = {"n": 0}
    original_create = completions.create

    def _create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 3:
            raise ConnectionError("simulated failure")
        return original_create(**kwargs)

    completions.create = _create

    class _FakeClient:
        def __init__(self, api_key=None, timeout=None, max_retries=None):
            self.chat = SimpleNamespace(completions=completions)

    monkeypatch.setattr("lab_core.lab_cli_support.Groq", _FakeClient)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")

    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab10", "all", "--live", "--profile", "pilot", "--experiment-run-id", "partial-fail", "--config", str(config_path)])
    args.func(args)  # must not raise

    with open(_run_dir(tmp_path, "partial-fail") / "followup_measurements.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6
    statuses = [r["request_status"] for r in rows]
    assert statuses.count("SUCCESS") == 5
    assert statuses.count("UNKNOWN_ERROR") == 1
