"""CLI wiring and orchestration for Lab06 -- Streaming & Perceived Latency.

Execution modes (the `experiment` positional argument) -- backward
compatible with the pre-redesign commands:
  - non_streaming / streaming: runs only that one condition, once per pair
    slot in `--profile`'s scope (no pairing partner).
  - all: both conditions, paired and interleaved -- this project's
    documented 10-call default (5 pairs x 2 conditions).
  - backfill: offline, read-only chart (re)generation from existing
    results -- see labs.lab06_streaming.charts_cli.

Every live-executing mode writes into its own
`results/lab06_streaming/<experiment_run_id>/` directory and never
touches the pre-existing flat `results/lab06_streaming.json`/`.csv` files
from before this redesign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import add_common_arguments, build_context
from labs.lab06_streaming.charts_cli import add_backfill_arguments, run_backfill
from labs.lab06_streaming.experiments import build_incident_messages
from labs.lab06_streaming.matrix import PROFILES, build_single_condition_plan, build_streaming_plan, pairs_for_profile
from labs.lab06_streaming.models import CONDITION_NON_STREAMING, CONDITION_STREAMING, StreamEventRecord, StreamMeasurement
from labs.lab06_streaming.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab06_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_condition_summary_csv,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
    write_paired_comparison_csv,
    write_stream_events_csv,
    write_stream_measurements,
)
from labs.lab06_streaming.report import build_analysis_report_markdown
from labs.lab06_streaming.runner import Budget, conservative_per_call_cost_ceiling, execute_planned_call
from labs.lab06_streaming.stats import compute_pair_comparisons, summarize_all_conditions

LAB_ID = "lab06"

_ALL_EXPERIMENTS = ["non_streaming", "streaming", "all", "backfill"]
_LONGEST_STATIC_PROMPT = build_incident_messages()[0]["content"]


def add_lab06_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Must be reasoning-capable (see config lab06.model).")
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Pair/strategy scope (default: config lab06.default_profile).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan, never call the API (always safe: "
        "ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab06 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab06-%Y%m%dT%H%M%SZ")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value:.1f}%" if value is not None else "N/A"


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _plan_for_experiment(experiment: str, pairs: int, randomize: bool, seed: int) -> list:
    if experiment == "all":
        return build_streaming_plan(pairs, randomize, seed)
    return build_single_condition_plan(experiment, pairs)


def _print_plan_header(model: str, experiment: str, pairs: int, total_planned: int, usage_requested: bool, live_enabled: bool) -> None:
    print("LAB06 -- Streaming & Perceived Latency\n")
    print(f"Model: {model}")
    print(f"Pairs: {pairs}")
    print(f"Planned API Calls: {total_planned}")
    print(f"Streaming Usage Requested: {usage_requested}\n")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_dry_run_preview(estimated_max_cost: float) -> None:
    print(f"\nEstimated Maximum Cost (conservative, pre-flight): {_fmt_cost(estimated_max_cost)}")


def _print_streaming_progress(m: StreamMeasurement, pairs: int) -> None:
    print(f"\nExperiment: streaming")
    print(f"Pair: {m.pair_id}/{pairs}\n")
    if not m.succeeded:
        print(f"FAILED: {m.error}")
        return
    print(f"First Stream Event:          {_fmt(m.first_stream_event_ms)} ms")
    print(f"First Visible Content:       {_fmt(m.first_visible_content_ms)} ms")
    print(f"First Event-to-Visible Gap:  {_fmt(m.first_event_to_visible_gap_ms)} ms")
    print(f"Last Visible Content:        {_fmt(m.last_visible_content_ms)} ms")
    print(f"Total Stream Latency:        {_fmt(m.total_stream_latency_ms)} ms")
    print(f"\nTotal Chunks:                {m.total_chunks}")
    print(f"Visible Content Chunks:      {m.visible_content_chunks}")
    print(f"Visible Characters:          {m.total_visible_characters}")
    print(f"\nPrompt Tokens:               {_fmt(m.prompt_tokens, 0)}")
    print(f"Completion Tokens:           {_fmt(m.completion_tokens, 0)}")
    print(f"Reasoning Tokens:            {_fmt(m.reasoning_tokens, 0)}")
    print(f"Token Usage Status:          {m.usage_status}")
    print(f"\nFinish Reason:               {m.finish_reason}")
    print(f"Answer Complete:             {m.answer_complete}")
    print(f"Evaluation Status:           {m.evaluation_status}")
    print(f"Estimated Cost:              {_fmt_cost(m.estimated_cost_usd)}")


def _print_non_streaming_progress(m: StreamMeasurement, pairs: int) -> None:
    print(f"\nExperiment: non_streaming")
    print(f"Pair: {m.pair_id}/{pairs}\n")
    if not m.succeeded:
        print(f"FAILED: {m.error}")
        return
    print(f"Total Client Latency:        {_fmt(m.non_streaming_total_latency_ms)} ms")
    print(f"Prompt Tokens:               {_fmt(m.prompt_tokens, 0)}")
    print(f"Completion Tokens:           {_fmt(m.completion_tokens, 0)}")
    print(f"Reasoning Tokens:            {_fmt(m.reasoning_tokens, 0)}")
    print(f"Effective Completion TPS:    {_fmt(m.effective_completion_tokens_per_second)}")
    print(f"Provider Generation TPS:     {_fmt(m.provider_completion_tokens_per_second)}")
    print(f"\nFinish Reason:               {m.finish_reason}")
    print(f"Answer Complete:             {m.answer_complete}")
    print(f"Estimated Cost:              {_fmt_cost(m.estimated_cost_usd)}")


def _print_pair_comparison(comparison) -> None:
    print(f"\n-- Pair {comparison.pair_id} Comparison --\n")
    print(f"Non-Streaming Full Response:  {_fmt(comparison.non_streaming_total_latency_ms)} ms")
    print(f"Streaming First Visible:      {_fmt(comparison.streaming_first_visible_content_ms)} ms")
    print(f"Streaming Total Latency:      {_fmt(comparison.streaming_total_latency_ms)} ms")
    print(f"\nPerceived Latency Gain:       {_fmt(comparison.perceived_latency_gain_ms)} ms ({_fmt_pct(comparison.perceived_latency_gain_percent)})")
    print(f"Total Latency Difference:     {_fmt(comparison.total_latency_difference_ms)} ms")
    print(f"\nQuality Comparison:           {comparison.quality_comparison}")


def _print_final_summary(summaries, comparisons) -> None:
    from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats

    by_condition = {s.condition: s for s in summaries}
    non_streaming = by_condition.get(CONDITION_NON_STREAMING)
    streaming = by_condition.get(CONDITION_STREAMING)

    print("\n-- LAB06 Grand Test Summary --\n")
    total_planned = sum(s.total_calls for s in summaries)
    total_successful = sum(s.successful_calls for s in summaries)
    print(f"Completed Pairs:               {len(comparisons)}")
    print(f"Successful Calls:              {total_successful}/{total_planned}")

    if streaming is not None:
        print(f"\nMean First Stream Event:       {_fmt(streaming.first_stream_event_ms.mean)} ms")
        print(f"Mean First Visible Content:    {_fmt(streaming.first_visible_content_ms.mean)} ms")
        print(f"Mean Streaming Total Latency:  {_fmt(streaming.total_latency_ms.mean)} ms")
    if non_streaming is not None:
        print(f"Mean Non-Streaming Latency:    {_fmt(non_streaming.total_latency_ms.mean)} ms")

    if comparisons:
        gains = [c.perceived_latency_gain_ms for c in comparisons if c.perceived_latency_gain_ms is not None]
        diffs = [c.total_latency_difference_ms for c in comparisons if c.total_latency_difference_ms is not None]
        gain_stats = compute_descriptive_stats(gains)
        diff_stats = compute_descriptive_stats(diffs)
        print(f"\nMean Perceived Latency Gain:   {_fmt(gain_stats.mean)} ms")
        print(f"Median Perceived Latency Gain: {_fmt(gain_stats.median)} ms")
        print(f"Mean Total Latency Difference: {_fmt(diff_stats.mean)} ms")

    if streaming is not None:
        print(f"\nStreaming Usage Available:     {streaming.usage_available_count}/{streaming.successful_calls}")
    if non_streaming is not None:
        print(f"Non-Streaming Usage Available: {non_streaming.usage_available_count}/{non_streaming.successful_calls}")

    completed = sum(s.completed_count for s in summaries)
    print(f"\nAnswer Completeness:           {completed}/{total_successful} COMPLETE")
    costs = [s.total_estimated_cost_usd for s in summaries if s.total_estimated_cost_usd is not None]
    print(f"Estimated Total Cost:          {_fmt_cost(sum(costs) if costs else None)}")

    if comparisons:
        assessment = "Streaming showed visible content earlier on average" if (compute_descriptive_stats([c.perceived_latency_gain_ms for c in comparisons if c.perceived_latency_gain_ms is not None]).mean or 0) > 0 else "No average perceived-latency advantage observed for streaming in this run"
        print(f"\nOverall Assessment:            {assessment}")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab06_config = probe_ctx.config.lab06
    if not lab06_config.enabled:
        print("Lab06 is disabled for this config (lab06.enabled = false).")
        return

    model = args.model or lab06_config.model
    profile = args.profile or lab06_config.default_profile
    pairs = args.repetitions or pairs_for_profile(profile, lab06_config.paired_repetitions)

    plan = _plan_for_experiment(args.experiment, pairs, lab06_config.randomize_condition_order, lab06_config.randomization_seed)
    total_planned = len(plan)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, args.experiment, pairs, total_planned, lab06_config.request_usage_when_supported, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab06_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        pricing = probe_ctx.config.pricing.get(model)
        estimated_max_cost = (
            conservative_per_call_cost_ceiling(pricing, lab06_config.max_completion_tokens, _LONGEST_STATIC_PROMPT) * total_planned
            if pricing is not None
            else 0.0
        )
        _print_dry_run_preview(estimated_max_cost)
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab06 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab06_config.max_api_calls is not None and total_planned > lab06_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab06.max_api_calls ({lab06_config.max_api_calls}). No API calls were made."
        )
        return

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    ctx = build_context(requested_live=True, config_path=args.config)
    pricing = ctx.config.pricing.get(model)
    if pricing is None:
        print(f"\nNo pricing configured for model '{model}': estimated_cost_usd will be None for every call.")

    per_call_ceiling = (
        conservative_per_call_cost_ceiling(pricing, lab06_config.max_completion_tokens, _LONGEST_STATIC_PROMPT)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=None,  # the pre-flight total_planned gate above already enforces this
        max_estimated_cost_usd=lab06_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )

    manifest = build_initial_manifest(experiment_run_id, lab06_config, profile, total_planned)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)

    measurements: list[StreamMeasurement] = []
    all_events: list[StreamEventRecord] = []
    stopped_reason = None

    for planned_call in plan:
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before call: {reason}")
            break

        measurement, events = execute_planned_call(
            ctx.client, ctx.config.retry, model, pricing, lab06_config.reasoning_effort, lab06_config.max_completion_tokens,
            lab06_config.request_usage_when_supported, lab06_config.record_event_timeline, experiment_run_id, planned_call,
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        measurements.append(measurement)
        all_events.extend(events)

        if measurement.condition == CONDITION_STREAMING:
            _print_streaming_progress(measurement, pairs)
        else:
            _print_non_streaming_progress(measurement, pairs)

    write_stream_measurements(run_dir, measurements)
    write_stream_events_csv(run_dir, all_events)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    write_manifest(run_dir, manifest)

    conditions_present = sorted({m.condition for m in measurements}, key=[CONDITION_NON_STREAMING, CONDITION_STREAMING].index)
    summaries = summarize_all_conditions(conditions_present, measurements)
    write_condition_summary_csv(run_dir, summaries)

    comparisons = compute_pair_comparisons(measurements)
    write_paired_comparison_csv(run_dir, comparisons)
    for comparison in comparisons:
        _print_pair_comparison(comparison)

    successes = [m for m in measurements if m.succeeded]
    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "profile": profile,
            "planned_calls": total_planned,
            "successful_calls": len(successes),
            "failed_calls": len(measurements) - len(successes),
            "stopped_reason": stopped_reason,
            "total_estimated_cost_usd": budget.accumulated_estimated_cost_usd,
            "conditions": [asdict(s) for s in summaries],
            "pair_comparisons": [asdict(c) for c in comparisons],
            "measurements": [asdict(m) for m in measurements],
            "stream_events": [asdict(e) for e in all_events],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, measurements, summaries, comparisons)
    write_analysis_report(run_dir, report_markdown)

    if successes:
        _print_final_summary(summaries, comparisons)
        # Best-effort, strictly after every measurement/report file above is
        # already written -- a charting failure must never cost an
        # already-persisted measurement.
        try:
            from labs.lab06_streaming.charts import generate_streaming_charts

            chart_paths = generate_streaming_charts(run_dir)
            if chart_paths:
                print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\n{'Experiment Completed' if stopped_reason is None else 'Experiment Stopped Early'}\n")
    if stopped_reason:
        print(f"Reason: {stopped_reason}\n")
    print(f"Planned Calls:     {total_planned}")
    print(f"Successful Calls:  {len(successes)}")
    print(f"Results Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab06 - Streaming & Perceived Latency")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="non_streaming/streaming run only that one condition; all runs both, paired and "
        "interleaved (this project's documented 10-call default); backfill (re)generates PNG "
        "charts from existing results, offline, without any API key.",
    )
    add_common_arguments(parser)
    add_lab06_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
