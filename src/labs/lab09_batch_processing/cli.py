"""CLI wiring and orchestration for Lab09 -- Batch Processing.

Execution modes (the `experiment` positional argument):
  - sequential/concurrent: run only that one real-time condition.
  - all: both real-time conditions (this project's documented 16-call
    default: 8 tasks x 2 conditions), plus writing (never submitting) the
    Batch input JSONL file. The Batch condition is explicitly left
    NOT_SUBMITTED -- actual provider submission is ALWAYS a separate,
    explicit command (spec Part 6), since it starts a deferred job that
    may run long after this command has returned.
  - submit_batch / check_batch_status / fetch_batch_results: the three
    stages of the Batch job lifecycle, each reading/writing the run's
    durable `batch_job.json` so they can be invoked as separate commands,
    possibly days apart, without resubmitting anything.
  - evaluate: recompute evaluation/comparisons from persisted measurements
    only -- never calls Groq.
  - backfill: offline chart (re)generation from existing results.

This lab constructs its OWN `AsyncGroq`/`Groq` clients (max_retries=
lab09.sdk_max_retries, default 0) -- never applied to any other lab's
shared client (see lab_core.lab_cli_support.build_context, left
untouched).
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import asdict
from datetime import datetime, timezone

from groq import AsyncGroq, Groq

from lab_core.lab_cli_support import LiveExecutionNotAllowed, add_common_arguments, build_context
from labs.lab09_batch_processing.analysis import compute_and_persist_summary
from labs.lab09_batch_processing.batch import (
    build_batch_requests,
    fetch_batch_job_fields,
    submit_batch_job,
    write_batch_jsonl,
)
from labs.lab09_batch_processing.batch import download_batch_file
from labs.lab09_batch_processing.charts_cli import add_backfill_arguments, run_backfill
from labs.lab09_batch_processing.datasets import (
    DatasetIntegrityError,
    PROFILES,
    check_dataset_integrity,
    load_dataset,
    task_ids_for_profile,
    tasks_by_id,
)
from labs.lab09_batch_processing.matrix import build_realtime_plan
from labs.lab09_batch_processing.models import (
    CONDITION_CONCURRENT,
    CONDITION_SEQUENTIAL,
    LOCAL_STATE_EVALUATED,
    LOCAL_STATE_NOT_SUBMITTED,
    LOCAL_STATE_RESULTS_AVAILABLE,
    LOCAL_STATE_RESULTS_IMPORTED,
    LOCAL_STATE_SUBMISSION_IN_PROGRESS,
    LOCAL_STATE_SUBMISSION_UNCERTAIN,
    LOCAL_STATE_SUBMITTED,
    BatchJobMetadata,
)
from labs.lab09_batch_processing.persistence import (
    BATCH_INPUT_FILENAME,
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab09_run_dir,
    read_batch_job,
    read_manifest,
    refuse_if_already_has_measurements,
    write_batch_job,
    write_batch_measurements,
    write_batch_raw_files,
    write_experiment_plan_csv,
    write_manifest,
    write_realtime_measurements,
    write_request_timeline_csv,
    write_workload_snapshot,
)
from labs.lab09_batch_processing.reconciliation import reconcile_batch_results
from labs.lab09_batch_processing.runner import run_realtime_batch

LAB_ID = "lab09"
_ALL_EXPERIMENTS = [
    "sequential", "concurrent", "all", "submit_batch", "check_batch_status",
    "fetch_batch_results", "evaluate", "backfill",
]


def add_lab09_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None)
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Task scope (default: config lab09.default_profile).")
    parser.add_argument("--max-concurrent", type=int, default=None, dest="max_concurrent", help="Override the concurrency level for the 'concurrent' condition.")
    parser.add_argument("--dry-run", action="store_true", help="Print and save the plan, never call the API (always safe).")
    parser.add_argument(
        "--experiment-run-id", default=None,
        help="Name this run (default: an auto-generated timestamp-based id for 'all'; REQUIRED for "
        "submit_batch/check_batch_status/fetch_batch_results/evaluate).",
    )
    # Backward-compatible fallbacks from the pre-2.0 CLI -- a detached, non-persisted
    # status check/fetch when no experiment-run-id/run directory is used.
    parser.add_argument("--batch-id", default=None, dest="batch_id")
    parser.add_argument("--output-file-id", default=None, dest="output_file_id")


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab09-%Y%m%dT%H%M%SZ")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _print_plan_header(model, profile, max_concurrent, total_planned, batch_requests, live_enabled) -> None:
    print("LAB09 -- Batch Processing (Sequential vs Concurrent vs Groq Batch API)\n")
    print(f"Model: {model}")
    print(f"Profile: {profile}\n")
    print(f"Concurrency (concurrent condition): {max_concurrent}")
    print(f"Planned Real-Time Calls: {total_planned}")
    print(f"Batch Input Requests (NOT additional real-time calls): {batch_requests}")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_condition_summary(name: str, summary) -> None:
    if summary is None:
        return
    print(f"\n-- {name} summary --")
    print(f"Successful Requests:     {summary.successful_requests}/{summary.total_requests}")
    print(f"Batch Wall-Clock:        {_fmt(summary.batch_wall_clock_ms)} ms")
    print(f"Successful Requests/sec: {_fmt(summary.successful_requests_per_second)}")
    print(f"Median Service Latency:  {_fmt(summary.request_service_latency_ms.median)} ms")
    print(f"Correct Answers:         {summary.correct_count}/{summary.successful_requests}")
    print(f"Total Estimated Cost:    {_fmt_cost(summary.total_estimated_cost_usd)}")


def _run_realtime(args: argparse.Namespace) -> None:
    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab09_config = probe_ctx.config.lab09
    if not lab09_config.enabled:
        print("Lab09 is disabled for this config (lab09.enabled = false).")
        return

    model = args.model or lab09_config.model
    profile = args.profile or lab09_config.default_profile
    max_concurrent = args.max_concurrent or lab09_config.max_concurrent_requests
    if max_concurrent < 1:
        print(f"\nRefusing: --max-concurrent must be >= 1 (got {max_concurrent}). Nothing was executed.")
        return

    try:
        tasks = load_dataset(lab09_config.dataset_version)
        check_dataset_integrity(tasks)
    except (DatasetIntegrityError, FileNotFoundError) as exc:
        print(f"\nGolden dataset failed its integrity check: {exc}. Nothing was executed.")
        return

    task_ids = task_ids_for_profile(tasks, profile)
    by_id = tasks_by_id(tasks)
    selected_tasks = [by_id[tid] for tid in task_ids]

    full_plan = build_realtime_plan(task_ids, max_concurrent)
    if args.experiment == "sequential":
        plan = [c for c in full_plan if c.condition == CONDITION_SEQUENTIAL]
    elif args.experiment == "concurrent":
        plan = [c for c in full_plan if c.condition == CONDITION_CONCURRENT]
    else:
        plan = full_plan
    total_planned = len(plan)
    batch_requests_count = len(task_ids)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, profile, max_concurrent, total_planned, batch_requests_count, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab09_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        write_workload_snapshot(run_dir, selected_tasks)
        pricing = probe_ctx.config.pricing.get(model)
        if pricing is not None:
            ceiling = total_planned * lab09_config.max_completion_tokens / 1_000_000 * pricing.output_per_1m_usd
            print(f"\nConservative Preflight Cost Ceiling (NOT a measured cost): ${ceiling:.8f}")
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab09 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab09_config.max_api_calls is not None and total_planned > lab09_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned real-time calls ({total_planned}) exceed "
            f"lab09.max_api_calls ({lab09_config.max_api_calls}). No API calls were made."
        )
        return

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    try:
        build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    pricing = probe_ctx.config.pricing.get(model)
    if pricing is None:
        print(f"\nNo pricing configured for model '{model}': estimated costs will be None for every call.")

    print(f"[COST WARNING] {probe_ctx.config.execution.cost_warning_message}")
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab09_config.sdk_max_retries)

    manifest = build_initial_manifest(experiment_run_id, lab09_config, profile, total_planned, batch_requests_count)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)
    write_workload_snapshot(run_dir, selected_tasks)

    sequential_calls = [c for c in plan if c.condition == CONDITION_SEQUENTIAL]
    concurrent_calls = [c for c in plan if c.condition == CONDITION_CONCURRENT]
    all_events = []

    if sequential_calls:
        seq_measurements, seq_events, seq_wall, seq_peak = asyncio.run(
            run_realtime_batch(client, model, pricing, lab09_config.reasoning_effort, lab09_config.max_completion_tokens, experiment_run_id, sequential_calls, by_id)
        )
        write_realtime_measurements(run_dir, "sequential", seq_measurements)
        all_events.extend(seq_events)
        manifest["sequential_wall_clock_ms"] = seq_wall
        manifest["sequential_peak_in_flight"] = seq_peak

    if concurrent_calls:
        conc_measurements, conc_events, conc_wall, conc_peak = asyncio.run(
            run_realtime_batch(client, model, pricing, lab09_config.reasoning_effort, lab09_config.max_completion_tokens, experiment_run_id, concurrent_calls, by_id)
        )
        write_realtime_measurements(run_dir, "concurrent", conc_measurements)
        all_events.extend(conc_events)
        manifest["concurrent_wall_clock_ms"] = conc_wall
        manifest["concurrent_peak_in_flight"] = conc_peak

    write_request_timeline_csv(run_dir, all_events)

    # Always write (never submit) the Batch input file for 'all' -- a Batch
    # job containing N input requests is not N additional real-time calls.
    if args.experiment == "all":
        batch_requests = build_batch_requests(selected_tasks, model)
        write_batch_jsonl(batch_requests, run_dir / BATCH_INPUT_FILENAME)
        if read_batch_job(run_dir) is None:
            write_batch_job(
                run_dir,
                BatchJobMetadata(
                    experiment_run_id=experiment_run_id, workload_version=lab09_config.dataset_version,
                    model=model, completion_window=lab09_config.batch_completion_window,
                ),
            )

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)

    summary_payload = compute_and_persist_summary(run_dir, lab09_config, model, pricing)

    if summary_payload.get("sequential_summary"):
        from labs.lab09_batch_processing.stats import RealtimeConditionSummary

        _print_condition_summary("Sequential", RealtimeConditionSummary(**_rehydrate_descriptive(summary_payload["sequential_summary"])))
    if summary_payload.get("concurrent_summary"):
        from labs.lab09_batch_processing.stats import RealtimeConditionSummary

        _print_condition_summary("Concurrent", RealtimeConditionSummary(**_rehydrate_descriptive(summary_payload["concurrent_summary"])))

    if args.experiment == "all":
        print(
            f"\nBatch condition: NOT_SUBMITTED. Submit it explicitly with:\n"
            f"  lab-cli lab09 submit_batch --experiment-run-id {experiment_run_id} --live"
        )

    try:
        from labs.lab09_batch_processing.charts import generate_batch_charts

        chart_paths = generate_batch_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001 -- a charting failure must never cost an already-persisted measurement.
        print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\nPlanned Real-Time Calls: {total_planned}")
    print(f"Batch Input Requests:    {batch_requests_count} (NOT_SUBMITTED)")
    print(f"Results Directory:       {run_dir}")


def _rehydrate_descriptive(payload: dict) -> dict:
    from labs.lab01_token_anatomy.factorial_stats import DescriptiveStats

    out = dict(payload)
    for key in ("queue_wait_ms", "request_service_latency_ms", "task_end_to_end_latency_ms"):
        if isinstance(out.get(key), dict):
            out[key] = DescriptiveStats(**out[key])
    return out


def _run_submit_batch(args: argparse.Namespace) -> None:
    if not args.experiment_run_id:
        print("\n--experiment-run-id is required for submit_batch. Nothing was executed.")
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab09_config = probe_ctx.config.lab09
    run_dir = lab09_run_dir(probe_ctx.output_dir, args.experiment_run_id)
    input_path = run_dir / BATCH_INPUT_FILENAME
    if not input_path.exists():
        print(f"\nNo batch input file at {input_path}. Run `lab09 all` (or 'prepare_batch_input') for this run-id first.")
        return

    job = read_batch_job(run_dir)
    if job is not None and job.local_lifecycle_state not in (LOCAL_STATE_NOT_SUBMITTED, LOCAL_STATE_SUBMISSION_UNCERTAIN):
        print(
            f"\nRefusing to submit again: this run's Batch job is already in state "
            f"'{job.local_lifecycle_state}' (batch_id={job.batch_id}). Lab09 never resubmits an "
            "already-submitted job -- use a different --experiment-run-id for a new job."
        )
        return
    if job is not None and job.local_lifecycle_state == LOCAL_STATE_SUBMISSION_UNCERTAIN:
        print(
            f"\n[WARNING] This run's previous submission attempt ended UNCERTAIN "
            f"(input_file_id={job.input_file_id}, last_known_error={job.last_known_error!r}). "
            "Reconcile manually against your Groq dashboard before resubmitting if in doubt."
        )

    model = args.model or lab09_config.model
    manifest = read_manifest(run_dir) or {}
    batch_requests_count = manifest.get("batch_input_requests")
    if lab09_config.max_batch_input_requests is not None and batch_requests_count is not None and batch_requests_count > lab09_config.max_batch_input_requests:
        print(
            f"\nRefusing to submit: batch input requests ({batch_requests_count}) exceed "
            f"lab09.max_batch_input_requests ({lab09_config.max_batch_input_requests}). Nothing was submitted."
        )
        return

    try:
        build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    print(f"[COST WARNING] {probe_ctx.config.execution.cost_warning_message}")
    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab09_config.sdk_max_retries)

    job = job or BatchJobMetadata(
        experiment_run_id=args.experiment_run_id, workload_version=lab09_config.dataset_version,
        model=model, completion_window=lab09_config.batch_completion_window,
    )
    job.local_lifecycle_state = LOCAL_STATE_SUBMISSION_IN_PROGRESS
    write_batch_job(run_dir, job)

    try:
        input_file_id, batch_id = submit_batch_job(client, input_path, lab09_config.batch_completion_window)
    except Exception as exc:  # noqa: BLE001 -- upload may have partially succeeded; record UNCERTAIN, never crash silently.
        job.local_lifecycle_state = LOCAL_STATE_SUBMISSION_UNCERTAIN
        job.last_known_error = f"{type(exc).__name__}: {exc}"
        write_batch_job(run_dir, job)
        print(f"\nSubmission ended UNCERTAIN: {exc}\nState persisted as SUBMISSION_UNCERTAIN -- see {run_dir / 'batch_job.json'}.")
        return

    job.input_file_id = input_file_id
    job.batch_id = batch_id
    job.local_lifecycle_state = LOCAL_STATE_SUBMITTED
    job.submitted_at = datetime.now(timezone.utc).isoformat()
    job.last_known_error = None
    write_batch_job(run_dir, job)

    print(f"\nSubmitted Batch job: {batch_id} (input_file_id={input_file_id})")
    print("This does NOT wait for completion (can take minutes up to the completion window, e.g. 24h-7d).")
    print(f"Check status later with: lab-cli lab09 check_batch_status --experiment-run-id {args.experiment_run_id} --live")


def _run_check_batch_status(args: argparse.Namespace) -> None:
    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab09_config = probe_ctx.config.lab09

    if args.experiment_run_id:
        run_dir = lab09_run_dir(probe_ctx.output_dir, args.experiment_run_id)
        job = read_batch_job(run_dir)
        if job is None or job.batch_id is None:
            print(f"\nNo submitted Batch job found for run '{args.experiment_run_id}'. Run `lab09 submit_batch` first.")
            return
        batch_id = job.batch_id
    elif args.batch_id:
        print("\n[DEPRECATED] --batch-id without --experiment-run-id is a detached check: nothing is persisted.")
        run_dir = None
        job = None
        batch_id = args.batch_id
    else:
        print("\n--experiment-run-id (or legacy --batch-id) is required for check_batch_status. Nothing was executed.")
        return

    try:
        build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab09_config.sdk_max_retries)
    fields = fetch_batch_job_fields(client, batch_id)
    print(f"\nBatch {batch_id}: status={fields['provider_status']} "
          f"request_counts(total={fields['request_counts_total']}, completed={fields['request_counts_completed']}, "
          f"failed={fields['request_counts_failed']}) output_file_id={fields['output_file_id']} error_file_id={fields['error_file_id']}")

    if job is not None and run_dir is not None:
        job.provider_status = fields["provider_status"]
        job.output_file_id = fields["output_file_id"]
        job.error_file_id = fields["error_file_id"]
        job.created_at = fields["created_at"]
        job.in_progress_at = fields["in_progress_at"]
        job.finalizing_at = fields["finalizing_at"]
        job.completed_at = fields["completed_at"]
        job.failed_at = fields["failed_at"]
        job.expired_at = fields["expired_at"]
        job.request_counts_total = fields["request_counts_total"]
        job.request_counts_completed = fields["request_counts_completed"]
        job.request_counts_failed = fields["request_counts_failed"]
        job.last_checked_at = datetime.now(timezone.utc).isoformat()
        if fields["provider_status"] == "completed" and job.local_lifecycle_state == LOCAL_STATE_SUBMITTED:
            job.local_lifecycle_state = LOCAL_STATE_RESULTS_AVAILABLE
        write_batch_job(run_dir, job)
        print(f"State persisted to {run_dir / 'batch_job.json'}.")


def _run_fetch_batch_results(args: argparse.Namespace) -> None:
    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab09_config = probe_ctx.config.lab09

    if not args.experiment_run_id:
        if args.output_file_id:
            print("\n[DEPRECATED] --output-file-id without --experiment-run-id is a detached fetch: results "
                  "are printed but never reconciled, evaluated, or persisted.")
            try:
                build_context(requested_live=True, config_path=args.config)
            except LiveExecutionNotAllowed as exc:
                print(f"\n{exc}")
                return
            client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab09_config.sdk_max_retries)
            rows = download_batch_file(client, args.output_file_id)
            print(f"Downloaded {len(rows)} raw output row(s). Re-run with --experiment-run-id for full reconciliation/evaluation.")
            return
        print("\n--experiment-run-id (or legacy --output-file-id) is required for fetch_batch_results. Nothing was executed.")
        return

    run_dir = lab09_run_dir(probe_ctx.output_dir, args.experiment_run_id)
    job = read_batch_job(run_dir)
    if job is None or job.batch_id is None:
        print(f"\nNo submitted Batch job found for run '{args.experiment_run_id}'. Run `lab09 submit_batch` first.")
        return

    manifest = read_manifest(run_dir) or {}
    tasks = load_dataset(lab09_config.dataset_version)
    by_id = tasks_by_id(tasks)
    task_ids = task_ids_for_profile(tasks, manifest.get("profile", lab09_config.default_profile))
    planned_tasks = [by_id[tid] for tid in task_ids]

    try:
        build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    client = Groq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab09_config.sdk_max_retries)
    status_fields = fetch_batch_job_fields(client, job.batch_id)
    output_file_id = status_fields["output_file_id"]
    error_file_id = status_fields["error_file_id"]

    if output_file_id is None and error_file_id is None:
        print(f"\nNo output/error file available yet (provider status: {status_fields['provider_status']}). "
              "BATCH_PENDING -- nothing to fetch.")
        return

    output_rows = download_batch_file(client, output_file_id)
    error_rows = download_batch_file(client, error_file_id)
    write_batch_raw_files(run_dir, output_rows, error_rows)

    model = args.model or job.model or lab09_config.model
    pricing = probe_ctx.config.pricing.get(model)
    measurements, counts = reconcile_batch_results(args.experiment_run_id, planned_tasks, output_rows, error_rows, model, pricing)
    write_batch_measurements(run_dir, measurements)

    job.provider_status = status_fields["provider_status"]
    job.output_file_id = output_file_id
    job.error_file_id = error_file_id
    job.request_counts_total = status_fields["request_counts_total"]
    job.request_counts_completed = status_fields["request_counts_completed"]
    job.request_counts_failed = status_fields["request_counts_failed"]
    job.completed_at = status_fields["completed_at"]
    job.local_lifecycle_state = LOCAL_STATE_EVALUATED
    job.last_checked_at = datetime.now(timezone.utc).isoformat()
    write_batch_job(run_dir, job)

    summary_payload = compute_and_persist_summary(run_dir, lab09_config, model, pricing)
    print(f"\nReconciled {counts.planned_requests} planned request(s): "
          f"{counts.completed_requests} completed, {counts.failed_requests} failed, {counts.missing_requests} missing.")
    print(f"Correct: {counts.correct_answers}, Incorrect: {counts.incorrect_answers}")
    if summary_payload.get("pricing_comparison"):
        pc = summary_payload["pricing_comparison"]
        print(f"Estimated Standard Cost: {_fmt_cost(pc['estimated_standard_cost_usd'])} | "
              f"Estimated Batch Cost: {_fmt_cost(pc['estimated_batch_cost_usd'])}")

    try:
        from labs.lab09_batch_processing.charts import generate_batch_charts

        chart_paths = generate_batch_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nChart generation failed (results above are unaffected): {exc}")


def _run_evaluate(args: argparse.Namespace) -> None:
    if not args.experiment_run_id:
        print("\n--experiment-run-id is required for evaluate. Nothing was executed.")
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab09_config = probe_ctx.config.lab09
    run_dir = lab09_run_dir(probe_ctx.output_dir, args.experiment_run_id)
    if not run_dir.is_dir():
        print(f"\nNo run directory found at {run_dir}. Nothing was executed.")
        return

    manifest = read_manifest(run_dir)
    if manifest is None:
        print(f"\nNo experiment_manifest.json found at {run_dir}. This run was never properly initialized. Nothing was executed.")
        return

    model = args.model or manifest.get("experiment_configuration", {}).get("model", lab09_config.model)
    pricing = probe_ctx.config.pricing.get(model)

    summary_payload = compute_and_persist_summary(run_dir, lab09_config, model, pricing)

    job = read_batch_job(run_dir)
    batch_state = job.local_lifecycle_state if job else "NOT_SUBMITTED"
    print(f"\nLAB09 -- Offline Evaluation (no API calls made)\n")
    print(f"Run: {args.experiment_run_id}")
    print(f"Sequential measured: {'yes' if summary_payload.get('sequential_summary') else 'no'}")
    print(f"Concurrent measured: {'yes' if summary_payload.get('concurrent_summary') else 'no'}")
    print(f"Batch lifecycle state: {batch_state}")
    if batch_state != LOCAL_STATE_EVALUATED:
        print("BATCH_PENDING -- Batch results are not yet available/imported for this run.")
    for q in summary_payload.get("quality_summaries", []):
        print(f"  {q['condition']}: {q['correct_count']}/{q['total_evaluated']} correct")

    try:
        from labs.lab09_batch_processing.charts import generate_batch_charts

        chart_paths = generate_batch_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001
        print(f"\nChart generation failed (results above are unaffected): {exc}")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return
    if args.experiment == "submit_batch":
        _run_submit_batch(args)
        return
    if args.experiment == "check_batch_status":
        _run_check_batch_status(args)
        return
    if args.experiment == "fetch_batch_results":
        _run_fetch_batch_results(args)
        return
    if args.experiment == "evaluate":
        _run_evaluate(args)
        return
    _run_realtime(args)


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(
        LAB_ID,
        help="Lab09 - Batch Processing (sequential vs concurrent vs Groq Batch API)",
    )
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="sequential/concurrent run one real-time condition; all runs both real-time conditions "
        "plus writing (never submitting) the Batch input file; submit_batch/check_batch_status/"
        "fetch_batch_results are the three explicit Batch lifecycle stages; evaluate recomputes "
        "offline from persisted measurements; backfill (re)generates charts offline.",
    )
    add_common_arguments(parser)
    add_lab09_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
