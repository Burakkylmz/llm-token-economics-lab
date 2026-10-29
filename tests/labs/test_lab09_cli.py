"""Offline, mocked end-to-end tests for the `lab09` CLI: dry-run safety,
live real-time execution (fake AsyncGroq client), the full Batch
lifecycle (fake sync Groq client: submit/status/fetch), budget and
duplicate-run-id refusal, offline `evaluate`/`backfill`, and that the
Batch condition is never auto-submitted by `all`.

Lab09 constructs its OWN AsyncGroq/Groq clients directly in
`labs.lab09_batch_processing.cli` (never the shared
`lab_core.lab_cli_support` client), so that module path is what gets
monkeypatched here.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from lab_core.cli import build_parser


def _usage(prompt_tokens=10, completion_tokens=2):
    return SimpleNamespace(
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, total_tokens=prompt_tokens + completion_tokens,
        total_time=0.01, prompt_tokens_details=None, completion_tokens_details=None,
    )


def _response(content="POSITIVE"):
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message, finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=_usage(), system_fingerprint=None)


class _FakeCompletions:
    def __init__(self):
        self.calls: list[dict] = []

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


class _FakeFiles:
    def __init__(self):
        self.created: list[dict] = []
        self.content_requests: list[str] = []
        self.content_by_id: dict[str, str] = {}

    def create(self, file, purpose):
        self.created.append({"purpose": purpose})
        return SimpleNamespace(id=f"file-{len(self.created)}")

    def content(self, file_id):
        self.content_requests.append(file_id)
        return SimpleNamespace(text=self.content_by_id.get(file_id, ""))


class _FakeBatches:
    def __init__(self):
        self.create_calls: list[dict] = []
        self.retrieve_responses: list[SimpleNamespace] = []

    def create(self, input_file_id, endpoint, completion_window):
        self.create_calls.append({"input_file_id": input_file_id, "endpoint": endpoint, "completion_window": completion_window})
        return SimpleNamespace(id="batch-123", status="validating")

    def retrieve(self, batch_id):
        if self.retrieve_responses:
            return self.retrieve_responses.pop(0)
        return SimpleNamespace(
            status="in_progress", output_file_id=None, error_file_id=None, created_at=1, in_progress_at=2,
            finalizing_at=None, completed_at=None, failed_at=None, expired_at=None, request_counts=None,
        )


def _make_fake_sync_groq_class():
    files = _FakeFiles()
    batches = _FakeBatches()
    construct_calls: list[dict] = []

    class _FakeGroq:
        def __init__(self, **kwargs):
            construct_calls.append(dict(kwargs))
            self.files = files
            self.batches = batches

    _FakeGroq.files = files
    _FakeGroq.batches = batches
    _FakeGroq.construct_calls = construct_calls
    return _FakeGroq


def _patch_groq(monkeypatch):
    fake_async = _make_fake_async_groq_class()
    fake_sync = _make_fake_sync_groq_class()
    monkeypatch.setattr("labs.lab09_batch_processing.cli.AsyncGroq", fake_async)
    monkeypatch.setattr("labs.lab09_batch_processing.cli.Groq", fake_sync)
    monkeypatch.setenv("GROQ_API_KEY", "test-key-not-real")
    return fake_async, fake_sync


def _fmt_toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, str):
        return f'"{v}"'
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_fmt_toml_value(x) for x in v) + "]"
    return str(v)


def _toml(tmp_path, live=True, **lab09_overrides) -> str:
    lab09_lines = "\n".join(f"{k} = {_fmt_toml_value(v)}" for k, v in lab09_overrides.items())
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
{lab09_lines}
"""


def _write_config(tmp_path, **kwargs):
    kwargs.setdefault("enabled", True)
    kwargs.setdefault("model", "openai/gpt-oss-20b")
    kwargs.setdefault("default_profile", "pilot")
    config_path = tmp_path / "config.toml"
    config_path.write_text(_toml(tmp_path, **kwargs), encoding="utf-8")
    return config_path


def _run_dir(tmp_path, run_id):
    return tmp_path / "results" / "lab09_batch_processing" / run_id


def test_dry_run_never_constructs_a_client_and_writes_plan(monkeypatch, tmp_path):
    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("dry-run must never construct a real client")

    monkeypatch.setattr("labs.lab09_batch_processing.cli.AsyncGroq", _ExplodingGroq)

    config_path = _write_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--dry-run", "--experiment-run-id", "dry-test", "--config", str(config_path)])
    args.func(args)

    run_dir = _run_dir(tmp_path, "dry-test")
    assert (run_dir / "experiment_plan.csv").exists()
    assert (run_dir / "golden_workload_snapshot.json").exists()
    assert not (run_dir / "sequential_measurements.json").exists()


def test_dry_run_pilot_profile_plans_eight_realtime_calls(tmp_path):
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--dry-run", "--profile", "pilot", "--experiment-run-id", "pilot-dry", "--config", str(config_path)])
    args.func(args)

    import csv

    with open(_run_dir(tmp_path, "pilot-dry") / "experiment_plan.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 8  # pilot: 4 tasks x 2 real-time conditions


def test_live_all_run_writes_batch_input_but_never_submits(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "live-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_async.completions.calls) == 8  # pilot: 4 tasks x 2 conditions
    assert len(fake_sync.batches.create_calls) == 0  # never auto-submitted

    run_dir = _run_dir(tmp_path, "live-test")
    assert (run_dir / "batch_input.jsonl").exists()
    with open(run_dir / "batch_job.json", encoding="utf-8") as f:
        job = json.load(f)
    assert job["local_lifecycle_state"] == "NOT_SUBMITTED"

    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["sequential_summary"]["successful_requests"] == 4
    assert summary["concurrent_summary"]["successful_requests"] == 4

    assert len(list((run_dir / "charts").glob("*.png"))) == 8


def test_sequential_command_runs_only_that_condition(monkeypatch, tmp_path):
    fake_async, _ = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "sequential", "--live", "--experiment-run-id", "seq-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_async.completions.calls) == 4
    run_dir = _run_dir(tmp_path, "seq-test")
    assert (run_dir / "sequential_measurements.json").exists()
    assert not (run_dir / "concurrent_measurements.json").exists()


def test_budget_refusal_blocks_execution(monkeypatch, tmp_path):
    fake_async, _ = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot", max_api_calls=4)
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "budget-test", "--config", str(config_path)])
    args.func(args)

    assert len(fake_async.completions.calls) == 0


def test_duplicate_run_id_is_refused(monkeypatch, tmp_path):
    fake_async, _ = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "sequential", "--live", "--experiment-run-id", "dup-test", "--config", str(config_path)])
    args.func(args)
    first_call_count = len(fake_async.completions.calls)

    args2 = parser.parse_args(["lab09", "sequential", "--live", "--experiment-run-id", "dup-test", "--config", str(config_path)])
    args2.func(args2)
    assert len(fake_async.completions.calls) == first_call_count  # refused, no new calls


def test_disabled_lab_makes_no_calls(monkeypatch, tmp_path):
    fake_async, _ = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, enabled=False, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "disabled-test", "--config", str(config_path)])
    args.func(args)
    assert len(fake_async.completions.calls) == 0


def test_live_flag_required_without_dry_run(tmp_path):
    config_path = _write_config(tmp_path, default_profile="pilot", live=False)
    parser = build_parser()
    args = parser.parse_args(["lab09", "all", "--experiment-run-id", "no-live-test", "--config", str(config_path)])
    args.func(args)  # must not raise -- just refuses and prints
    assert not _run_dir(tmp_path, "no-live-test").exists() or not (_run_dir(tmp_path, "no-live-test") / "sequential_measurements.json").exists()


def test_submit_batch_requires_prior_all_run(monkeypatch, tmp_path):
    _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "never-ran", "--live", "--config", str(config_path)])
    args.func(args)  # must not raise -- prints a clear message, no crash


def test_submit_batch_writes_submitted_state(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "submit-test", "--config", str(config_path)])
    all_args.func(all_args)

    submit_args = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "submit-test", "--live", "--config", str(config_path)])
    submit_args.func(submit_args)

    assert len(fake_sync.batches.create_calls) == 1
    with open(_run_dir(tmp_path, "submit-test") / "batch_job.json", encoding="utf-8") as f:
        job = json.load(f)
    assert job["local_lifecycle_state"] == "SUBMITTED"
    assert job["batch_id"] == "batch-123"


def test_submit_batch_never_resubmits_an_already_submitted_job(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "no-resubmit-test", "--config", str(config_path)])
    all_args.func(all_args)
    submit_args = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "no-resubmit-test", "--live", "--config", str(config_path)])
    submit_args.func(submit_args)
    assert len(fake_sync.batches.create_calls) == 1

    submit_args_again = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "no-resubmit-test", "--live", "--config", str(config_path)])
    submit_args_again.func(submit_args_again)
    assert len(fake_sync.batches.create_calls) == 1  # refused, no second job created


def test_check_batch_status_persists_updated_fields(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "status-test", "--config", str(config_path)])
    all_args.func(all_args)
    submit_args = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "status-test", "--live", "--config", str(config_path)])
    submit_args.func(submit_args)

    fake_sync.batches.retrieve_responses.append(
        SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id=None, created_at=1, in_progress_at=2,
            finalizing_at=3, completed_at=4, failed_at=None, expired_at=None,
            request_counts=SimpleNamespace(total=4, completed=4, failed=0),
        )
    )
    status_args = parser.parse_args(["lab09", "check_batch_status", "--experiment-run-id", "status-test", "--live", "--config", str(config_path)])
    status_args.func(status_args)

    with open(_run_dir(tmp_path, "status-test") / "batch_job.json", encoding="utf-8") as f:
        job = json.load(f)
    assert job["provider_status"] == "completed"
    assert job["request_counts_completed"] == 4
    assert job["local_lifecycle_state"] == "RESULTS_AVAILABLE"


def test_fetch_batch_results_reconciles_and_evaluates(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "fetch-test", "--config", str(config_path)])
    all_args.func(all_args)
    submit_args = parser.parse_args(["lab09", "submit_batch", "--experiment-run-id", "fetch-test", "--live", "--config", str(config_path)])
    submit_args.func(submit_args)

    run_dir = _run_dir(tmp_path, "fetch-test")
    with open(run_dir / "batch_input.jsonl", encoding="utf-8") as f:
        input_lines = [json.loads(line) for line in f if line.strip()]
    output_lines = []
    for line in input_lines:
        output_lines.append(json.dumps({
            "custom_id": line["custom_id"],
            "response": {"status_code": 200, "body": {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}, "choices": [{"message": {"content": "POSITIVE"}}]}},
            "error": None,
        }))
    fake_sync.files.content_by_id["out-1"] = "\n".join(output_lines)
    fake_sync.batches.retrieve_responses.append(
        SimpleNamespace(
            status="completed", output_file_id="out-1", error_file_id=None, created_at=1, in_progress_at=2,
            finalizing_at=3, completed_at=4, failed_at=None, expired_at=None,
            request_counts=SimpleNamespace(total=4, completed=4, failed=0),
        )
    )

    fetch_args = parser.parse_args(["lab09", "fetch_batch_results", "--experiment-run-id", "fetch-test", "--live", "--config", str(config_path)])
    fetch_args.func(fetch_args)

    assert (run_dir / "batch_output.jsonl").exists()
    assert (run_dir / "batch_measurements.json").exists()
    with open(run_dir / "experiment_summary.json", encoding="utf-8") as f:
        summary = json.load(f)
    assert summary["batch_counts"]["completed_requests"] == 4
    assert summary["pricing_comparison"]["estimated_batch_cost_usd"] is not None
    with open(run_dir / "batch_job.json", encoding="utf-8") as f:
        job = json.load(f)
    assert job["local_lifecycle_state"] == "EVALUATED"


def test_evaluate_never_calls_groq_and_reuses_persisted_data(monkeypatch, tmp_path):
    fake_async, fake_sync = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "eval-test", "--config", str(config_path)])
    all_args.func(all_args)
    call_count_before = len(fake_async.completions.calls)

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("evaluate must never construct a Groq client")

    monkeypatch.setattr("labs.lab09_batch_processing.cli.AsyncGroq", _ExplodingGroq)
    monkeypatch.setattr("labs.lab09_batch_processing.cli.Groq", _ExplodingGroq)

    eval_args = parser.parse_args(["lab09", "evaluate", "--experiment-run-id", "eval-test", "--config", str(config_path)])
    eval_args.func(eval_args)

    assert len(fake_async.completions.calls) == call_count_before  # unchanged -- no new API calls


def test_evaluate_without_experiment_run_id_fails_clearly(tmp_path):
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "evaluate", "--config", str(config_path)])
    args.func(args)  # must not raise -- prints a clear message


def test_evaluate_on_nonexistent_run_fails_clearly(tmp_path):
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    args = parser.parse_args(["lab09", "evaluate", "--experiment-run-id", "does-not-exist", "--config", str(config_path)])
    args.func(args)  # must not raise -- prints a clear message


def test_backfill_regenerates_charts_offline(monkeypatch, tmp_path):
    fake_async, _ = _patch_groq(monkeypatch)
    config_path = _write_config(tmp_path, default_profile="pilot")
    parser = build_parser()
    all_args = parser.parse_args(["lab09", "all", "--live", "--experiment-run-id", "backfill-test", "--config", str(config_path)])
    all_args.func(all_args)

    run_dir = _run_dir(tmp_path, "backfill-test")
    import shutil

    shutil.rmtree(run_dir / "charts")
    assert not (run_dir / "charts").exists()

    class _ExplodingGroq:
        def __init__(self, *a, **k):
            raise AssertionError("backfill must never construct a Groq client")

    monkeypatch.setattr("labs.lab09_batch_processing.cli.AsyncGroq", _ExplodingGroq)

    backfill_args = parser.parse_args(["lab09", "backfill", "--run-id", "backfill-test", "--config", str(config_path)])
    backfill_args.func(backfill_args)

    assert len(list((run_dir / "charts").glob("*.png"))) == 8
