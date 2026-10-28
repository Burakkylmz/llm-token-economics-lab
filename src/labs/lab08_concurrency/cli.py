"""CLI wiring and orchestration for Lab08 -- Async Processing &
Concurrency.

Execution modes (the `experiment` positional argument) -- backward
compatible with the pre-redesign commands:
  - sequential: runs only concurrency=1 (still via the async client and
    this lab's own instrumented engine -- `execution_mode` is derived
    from `max_concurrent`, never hardcoded).
  - concurrent: runs only one concurrency level (`--max-concurrent`
    override, else the highest level `--profile` enables).
  - all: every concurrency level `--profile` enables -- this project's
    documented 16-call default (8 tasks x 2 levels: 1, 5).
  - backfill: offline chart (re)generation from existing results.

This lab constructs its OWN `AsyncGroq` client with `max_retries=
lab08.sdk_max_retries` (default 0) -- never applied to any other lab's
shared client (see lab_core.lab_cli_support.build_context, left
untouched).
"""

from __future__ import annotations

import argparse
import asyncio
import itertools
import os
from dataclasses import asdict
from datetime import datetime, timezone

from groq import AsyncGroq

from lab_core.lab_cli_support import LiveExecutionNotAllowed, add_common_arguments, build_context
from labs.lab08_concurrency.charts_cli import add_backfill_arguments, run_backfill
from labs.lab08_concurrency.matrix import PROFILES, build_plan, levels_for_profile, task_ids_for_profile
from labs.lab08_concurrency.models import ConcurrencyMeasurement, PlannedConcurrencyCall, RequestTimelineEvent
from labs.lab08_concurrency.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab08_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_concurrency_comparison_csv,
    write_condition_summary_csv,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
    write_raw_measurements,
    write_request_timeline_csv,
)
from labs.lab08_concurrency.report import build_analysis_report_markdown
from labs.lab08_concurrency.runner import run_concurrency_level
from labs.lab08_concurrency.stats import compare_to_baseline, summarize_all_conditions

LAB_ID = "lab08"
_ALL_EXPERIMENTS = ["sequential", "concurrent", "all", "backfill"]


def add_lab08_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Must be reasoning-capable (see config lab08.model).")
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Task/concurrency scope (default: config lab08.default_profile).")
    parser.add_argument("--max-concurrent", type=int, default=None, dest="max_concurrent", help="Override the concurrency level for 'concurrent' (and 'sequential', where it is ignored).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan, never call the API (always safe: "
        "ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id", default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab08 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab08-%Y%m%dT%H%M%SZ")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _group_into_batches(plan: list[PlannedConcurrencyCall]) -> list[list[PlannedConcurrencyCall]]:
    return [list(g) for _, g in itertools.groupby(plan, key=lambda c: (c.repetition_index, c.max_concurrent))]


def _levels_for_experiment(experiment: str, profile: str, default_levels, max_concurrent_override) -> tuple[int, ...]:
    profile_levels = levels_for_profile(profile, default_levels)
    if experiment == "sequential":
        return (1,)
    if experiment == "concurrent":
        return (max_concurrent_override or profile_levels[-1],)
    return profile_levels


def _print_plan_header(model, profile, levels, repetitions, total_planned, live_enabled) -> None:
    print("LAB08 -- Async Processing & Concurrency\n")
    print(f"Model: {model}")
    print(f"Profile: {profile}\n")
    print(f"Concurrency Levels: {list(levels)}")
    print(f"Repetitions: {repetitions}\n")
    print(f"Planned API Calls: {total_planned}")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_request_progress(m: ConcurrencyMeasurement) -> None:
    print(
        f"  [{m.execution_index}] {m.task_id} concurrency={m.max_concurrent} status={m.request_status} "
        f"queue_wait={_fmt(m.queue_wait_ms)}ms service={_fmt(m.request_service_latency_ms)}ms "
        f"e2e={_fmt(m.task_end_to_end_latency_ms)}ms cost={_fmt_cost(m.estimated_cost_usd)}"
    )


def _print_batch_summary(level: int, measurements, wall_clock_ms, peak) -> None:
    from labs.lab08_concurrency.stats import summarize_condition

    summary = summarize_condition(level, measurements, [], wall_clock_ms, peak)
    print(f"\n-- concurrency={level} summary --\n")
    print(f"Completed Requests:           {summary.total_requests}/{summary.total_requests}")
    print(f"Successful Requests:          {summary.successful_requests}/{summary.total_requests}")
    print(f"\nConfigured Concurrency:       {level}")
    print(f"Observed Peak Concurrency:    {summary.observed_peak_in_flight_requests}")
    print(f"\nBatch Wall-Clock Time:        {_fmt(summary.batch_wall_clock_ms)} ms")
    print(f"Successful Requests/sec:      {_fmt(summary.successful_requests_per_second)}")
    print(f"\nMean Queue Wait:              {_fmt(summary.queue_wait_ms.mean)} ms")
    print(f"Median Service Latency:       {_fmt(summary.request_service_latency_ms.median)} ms")
    print(f"Median End-to-End Latency:    {_fmt(summary.task_end_to_end_latency_ms.median)} ms")
    print(f"P95 End-to-End Latency:       {_fmt(summary.task_end_to_end_latency_ms.p95)} ms")
    print(f"\nAccuracy:                     {summary.correct_count}/{summary.successful_requests}")
    print(f"Total Estimated Cost:         {_fmt_cost(summary.total_estimated_cost_usd)}")
    print(f"Rate-Limited Requests:        {summary.rate_limited_requests}")


def _print_grand_summary(summaries, comparisons) -> None:
    print("\n-- LAB08 Grand Test Summary --\n")
    for s in summaries:
        print(f"Concurrency={s.max_concurrent} ({s.execution_mode}):")
        print(f"  Batch Wall-Clock:          {_fmt(s.batch_wall_clock_ms)} ms")
        print(f"  Successful RPS:            {_fmt(s.successful_requests_per_second)}")
        print(f"  P50 Service Latency:       {_fmt(s.request_service_latency_ms.median)} ms")
        print(f"  P95 Service Latency:       {_fmt(s.request_service_latency_ms.p95)} ms")
        print(f"  Success Rate:              {_fmt_pct(s.success_rate)}\n")

    for c in comparisons:
        print(f"-- Comparison: concurrency={c.baseline_concurrency} -> concurrency={c.compared_concurrency} --")
        print(f"  Throughput Speedup:        {_fmt(c.throughput_speedup_ratio)}x")
        print(f"  Batch Time Reduction:      {_fmt_pct(c.batch_time_reduction_percent)}")
        print(f"  Service Latency Change:    {_fmt_pct(c.median_service_latency_delta_percent)}")
        print(f"  End-to-End Latency Change: {_fmt_pct(c.median_task_end_to_end_latency_delta_percent)}")
        print(f"  Success Rate Delta:        {_fmt(c.success_rate_delta)}")
        print(f"  Cost Delta:                {_fmt_pct(c.cost_delta_percent)}\n")


async def _execute_all_batches(client, model, pricing, reasoning_effort, max_completion_tokens, experiment_run_id, plan):
    plan_results = []
    all_measurements: list[ConcurrencyMeasurement] = []
    all_events: list[RequestTimelineEvent] = []

    for batch_calls in _group_into_batches(plan):
        level = batch_calls[0].max_concurrent
        measurements, events, wall_clock_ms, peak = await run_concurrency_level(
            client, model, pricing, reasoning_effort, max_completion_tokens, experiment_run_id, batch_calls
        )
        for m in measurements:
            _print_request_progress(m)
        _print_batch_summary(level, measurements, wall_clock_ms, peak)

        all_measurements.extend(measurements)
        all_events.extend(events)
        plan_results.append((level, measurements, events, wall_clock_ms, peak))

    return all_measurements, all_events, plan_results


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab08_config = probe_ctx.config.lab08
    if not lab08_config.enabled:
        print("Lab08 is disabled for this config (lab08.enabled = false).")
        return

    model = args.model or lab08_config.model
    profile = args.profile or lab08_config.default_profile
    repetitions = args.repetitions or lab08_config.repetitions

    if args.max_concurrent is not None and args.max_concurrent < 1:
        print(f"\nRefusing: --max-concurrent must be >= 1 (got {args.max_concurrent}). Nothing was executed.")
        return

    levels = _levels_for_experiment(args.experiment, profile, lab08_config.concurrency_levels, args.max_concurrent)
    for level in levels:
        if level > lab08_config.max_allowed_concurrency:
            print(
                f"\nRefusing: concurrency {level} exceeds lab08.max_allowed_concurrency "
                f"({lab08_config.max_allowed_concurrency}). Nothing was executed."
            )
            return

    task_ids = task_ids_for_profile(profile)
    plan = build_plan(task_ids, levels, repetitions, lab08_config.randomize_condition_order, lab08_config.randomization_seed)
    total_planned = len(plan)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, profile, levels, repetitions, total_planned, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab08_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab08 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab08_config.max_api_calls is not None and total_planned > lab08_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab08.max_api_calls ({lab08_config.max_api_calls}). No API calls were made."
        )
        return

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    # Validate the live-execution guard (API key presence, live_api_enabled)
    # via the shared helper, but construct THIS lab's own AsyncGroq client
    # below so sdk_max_retries is never applied to any other lab's client.
    try:
        build_context(requested_live=True, config_path=args.config)
    except LiveExecutionNotAllowed as exc:
        print(f"\n{exc}")
        return

    pricing = probe_ctx.config.pricing.get(model)
    if pricing is None:
        print(f"\nNo pricing configured for model '{model}': estimated_cost_usd will be None for every call.")

    print(f"[COST WARNING] {probe_ctx.config.execution.cost_warning_message}")
    client = AsyncGroq(api_key=os.environ["GROQ_API_KEY"], max_retries=lab08_config.sdk_max_retries)

    manifest = build_initial_manifest(experiment_run_id, lab08_config, profile, total_planned)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)

    all_measurements, all_events, plan_results = asyncio.run(
        _execute_all_batches(client, model, pricing, lab08_config.reasoning_effort, lab08_config.max_completion_tokens, experiment_run_id, plan)
    )

    write_raw_measurements(run_dir, all_measurements)
    write_request_timeline_csv(run_dir, all_events)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)

    summaries = summarize_all_conditions(plan_results)
    write_condition_summary_csv(run_dir, summaries)

    by_level = {s.max_concurrent: s for s in summaries}
    baseline_level = min(by_level.keys())
    comparisons = [compare_to_baseline(by_level[baseline_level], s) for s in summaries if s.max_concurrent != baseline_level]
    write_concurrency_comparison_csv(run_dir, comparisons)

    successes = [m for m in all_measurements if m.request_status == "SUCCESS"]
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "profile": profile,
            "planned_calls": total_planned,
            "successful_calls": len(successes),
            "failed_calls": len(all_measurements) - len(successes),
            "total_estimated_cost_usd": sum(costs) if costs else None,
            "conditions": [asdict(s) for s in summaries],
            "comparisons": [asdict(c) for c in comparisons],
            "measurements": [asdict(m) for m in all_measurements],
            "timeline_events": [asdict(e) for e in all_events],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, all_measurements, summaries, comparisons)
    write_analysis_report(run_dir, report_markdown)

    _print_grand_summary(summaries, comparisons)

    try:
        from labs.lab08_concurrency.charts import generate_concurrency_charts

        chart_paths = generate_concurrency_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001 -- a charting failure must never cost an already-persisted measurement.
        print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\nPlanned Calls:     {total_planned}")
    print(f"Successful Calls:  {len(successes)}")
    print(f"Results Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab08 - Async Processing & Concurrency")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="sequential runs only concurrency=1; concurrent runs only one level (--max-concurrent "
        "or the profile's highest level); all runs every level --profile enables (this project's "
        "documented 16-call default: 8 tasks x 2 levels); backfill (re)generates PNG charts from "
        "existing results, offline, without any API key.",
    )
    add_common_arguments(parser)
    add_lab08_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
