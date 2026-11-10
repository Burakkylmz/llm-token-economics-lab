"""Offline tests for automatic chart generation in Lab01's basic execution
modes: `lab01 short_prompt` / `long_prompt` / `all`.

Verifies the requirement that a student gets BOTH measurements (JSON/CSV)
AND charts (PNG) from one single CLI/Docker invocation, with no separate
`charts`/`backfill` command needed, no --dry-run measurement charting, no
extra API calls, and no measurement loss if chart generation itself fails.

No real Groq API calls anywhere: basic mode's `--live` path is exercised
via a fake Groq client monkeypatched in place of lab_core.lab_cli_support's
real client classes, the same pattern used by test_lab01_effort.py.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lab_core.cli import build_parser
from lab_core.metrics import DerivedMetrics, ExecutionMetadata, ExperimentResult, PerformanceMetrics, TokenCostMetrics
from labs.lab01_token_anatomy.basic_charts import generate_basic_mode_charts


def _make_result(i, completion, reasoning, finish="stop"):
    non_reasoning = completion - reasoning if reasoning is not None and completion is not None else None
    return ExperimentResult(
        metadata=ExecutionMetadata(
            experiment_id="lab01_short_prompt", run_id=str(i), model_id="openai/gpt-oss-20b",
            execution_mode="sync", timestamp="2026-10-27T00:00:00+00:00", run_index=i,
        ),
        tokens_and_cost=TokenCostMetrics(
            prompt_tokens=10, completion_tokens=completion, total_tokens=(10 + completion) if completion else None,
            reasoning_tokens=reasoning, estimated_cost_usd=0.0001, finish_reason=finish,
        ),
        performance=PerformanceMetrics(),
        derived=DerivedMetrics(non_reasoning_completion_tokens=non_reasoning),
    )


class TestGenerateBasicModeCharts:
    def test_single_experiment_writes_four_charts(self, tmp_path):
        results = [_make_result(i, 20 + i, 5 + i) for i in range(1, 6)]
        written = generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", {"short_prompt": results})
        assert len(written) == 4
        for path in written:
            assert path.exists() and path.stat().st_size > 0
            assert path.parent.name == "charts"
            assert path.parent.parent.name == "lab01_token_anatomy"

    def test_all_mode_facets_short_and_long_separately_not_averaged(self, tmp_path):
        results_by_experiment = {
            "short_prompt": [_make_result(i, 20 + i, 5) for i in range(1, 6)],
            "long_prompt": [_make_result(i, 900 + i, 100) for i in range(1, 6)],
        }
        written = generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", results_by_experiment)
        # 4 charts x 2 experiments = 8, each experiment's own filename prefix.
        assert len(written) == 8
        names = {p.name for p in written}
        assert "short_prompt_dashboard.png" in names
        assert "long_prompt_dashboard.png" in names

    def test_empty_results_dict_returns_empty(self, tmp_path):
        assert generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", {}) == []

    def test_experiment_with_empty_results_list_is_skipped(self, tmp_path):
        assert generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", {"short_prompt": []}) == []

    def test_missing_reasoning_tokens_not_fabricated_as_zero(self, tmp_path):
        # No call in this run reported reasoning_tokens at all (e.g. a
        # non-reasoning model) -- the reasoning-vs-non-reasoning panel must
        # not crash and must not draw a fabricated zero bar.
        results = [_make_result(i, 20 + i, None) for i in range(1, 4)]
        written = generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", {"short_prompt": results})
        assert len(written) == 4  # still writes all 4 files; the one panel just renders "no data"

    def test_single_run_does_not_crash_and_still_produces_charts(self, tmp_path):
        results = [_make_result(1, 25, 6)]
        written = generate_basic_mode_charts(tmp_path, "lab01_token_anatomy", {"short_prompt": results})
        assert len(written) == 4


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


def _toml(tmp_path, live=True) -> str:
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
"""


class TestBasicModeCliAutoChartGeneration:
    def test_short_prompt_live_run_writes_measurements_and_charts_in_one_command(self, monkeypatch, tmp_path, make_response):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path), encoding="utf-8")
        _patch_groq(monkeypatch, [make_response(completion_tokens=20, reasoning_tokens=5) for _ in range(2)])

        parser = build_parser()
        args = parser.parse_args(["lab01", "short_prompt", "--live", "--config", str(config_path)])
        args.func(args)

        results_dir = tmp_path / "results"
        assert (results_dir / "lab01_token_anatomy.json").exists()
        assert (results_dir / "lab01_token_anatomy.csv").exists()
        assert (results_dir / "lab01_token_anatomy_summary.json").exists()

        charts_dir = results_dir / "lab01_token_anatomy" / "charts"
        assert charts_dir.is_dir()
        png_files = list(charts_dir.glob("*.png"))
        assert len(png_files) == 4
        for png in png_files:
            assert png.stat().st_size > 0

    def test_long_prompt_live_run_auto_generates_charts(self, monkeypatch, tmp_path, make_response):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path), encoding="utf-8")
        _patch_groq(monkeypatch, [make_response(completion_tokens=900, reasoning_tokens=100) for _ in range(2)])

        parser = build_parser()
        args = parser.parse_args(["lab01", "long_prompt", "--live", "--config", str(config_path)])
        args.func(args)

        charts_dir = tmp_path / "results" / "lab01_token_anatomy" / "charts"
        assert charts_dir.is_dir()
        assert len(list(charts_dir.glob("long_prompt_*.png"))) == 4

    def test_all_mode_auto_generates_charts_for_both_prompts_separately(self, monkeypatch, tmp_path, make_response):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path), encoding="utf-8")
        # 2 repetitions x 2 experiments (short_prompt, long_prompt) = 4 calls.
        _patch_groq(monkeypatch, [make_response(completion_tokens=20, reasoning_tokens=5) for _ in range(4)])

        parser = build_parser()
        args = parser.parse_args(["lab01", "all", "--live", "--config", str(config_path)])
        args.func(args)

        charts_dir = tmp_path / "results" / "lab01_token_anatomy" / "charts"
        assert len(list(charts_dir.glob("short_prompt_*.png"))) == 4
        assert len(list(charts_dir.glob("long_prompt_*.png"))) == 4

    def test_without_live_flag_no_api_call_and_no_charts(self, monkeypatch, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path, live=False), encoding="utf-8")

        client_constructed = {"called": False}

        class _ExplodingGroq:
            def __init__(self, *a, **k):
                client_constructed["called"] = True
                raise AssertionError("must never construct a real client without --live")

        monkeypatch.setattr("lab_core.lab_cli_support.Groq", _ExplodingGroq)
        monkeypatch.setattr("lab_core.lab_cli_support.AsyncGroq", _ExplodingGroq)

        parser = build_parser()
        args = parser.parse_args(["lab01", "short_prompt", "--config", str(config_path)])
        args.func(args)

        assert client_constructed["called"] is False
        assert not (tmp_path / "results").exists()

    def test_chart_generation_failure_does_not_lose_measurements(self, monkeypatch, tmp_path, make_response, capsys):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path), encoding="utf-8")
        _patch_groq(monkeypatch, [make_response(completion_tokens=20, reasoning_tokens=5) for _ in range(2)])

        def _boom(*args, **kwargs):
            raise RuntimeError("simulated chart failure")

        monkeypatch.setattr("labs.lab01_token_anatomy.basic_charts.generate_basic_mode_charts", _boom)

        parser = build_parser()
        args = parser.parse_args(["lab01", "short_prompt", "--live", "--config", str(config_path)])
        args.func(args)  # must not raise

        results_dir = tmp_path / "results"
        assert (results_dir / "lab01_token_anatomy.json").exists()
        assert (results_dir / "lab01_token_anatomy.csv").exists()
        assert (results_dir / "lab01_token_anatomy_summary.json").exists()
        assert "Chart generation failed" in capsys.readouterr().out

    def test_charts_do_not_collide_with_sibling_json_csv_files(self, monkeypatch, tmp_path, make_response):
        config_path = tmp_path / "config.toml"
        config_path.write_text(_toml(tmp_path), encoding="utf-8")
        _patch_groq(monkeypatch, [make_response(completion_tokens=20, reasoning_tokens=5) for _ in range(2)])

        parser = build_parser()
        args = parser.parse_args(["lab01", "short_prompt", "--live", "--config", str(config_path)])
        args.func(args)

        results_dir = tmp_path / "results"
        # lab01_token_anatomy.json (file) and lab01_token_anatomy/ (dir) coexist.
        assert (results_dir / "lab01_token_anatomy.json").is_file()
        assert (results_dir / "lab01_token_anatomy").is_dir()
