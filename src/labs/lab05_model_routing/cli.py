"""CLI wiring and orchestration for Lab05 -- Model Routing.

Execution modes (the `experiment` positional argument), all evaluated over
`--profile`'s task set (pilot=4 tasks, default=6 tasks, extended=6 tasks):
  - fixed_model_baseline: Strategy A (Always-Small) only -- the direct,
    formalized replacement for this command's pre-redesign behavior.
  - rule_based_routing: Strategy C (Rule-Based Routing) only.
  - always_large: Strategy B (Always-Large) only.
  - all: every strategy `--profile` enables (pilot/default: Always-Small +
    Router = this project's documented 8/12-call totals; extended: all
    three strategies = 18 calls).
  - backfill: offline, read-only chart (re)generation from existing
    results -- see labs.lab05_model_routing.charts_cli.

Every live-executing mode writes into its own
`results/lab05_model_routing/<experiment_run_id>/` directory and never
touches the pre-existing flat `results/lab05_model_routing.json`/`.csv`
files from before this redesign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import add_common_arguments, build_context
from labs.lab05_model_routing.charts_cli import add_backfill_arguments, run_backfill
from labs.lab05_model_routing.golden_dataset import TASK_BY_ID
from labs.lab05_model_routing.matrix import (
    PROFILES,
    STRATEGY_ALWAYS_LARGE,
    STRATEGY_ALWAYS_SMALL,
    STRATEGY_ROUTER,
    build_plan_for_profile,
    build_routing_plan,
    strategies_for_profile,
    task_ids_for,
)
from labs.lab05_model_routing.models import RoutingMeasurement
from labs.lab05_model_routing.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab05_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
    write_routing_measurements,
    write_strategy_summary_csv,
    write_task_comparison_csv,
)
from labs.lab05_model_routing.report import build_analysis_report_markdown
from labs.lab05_model_routing.runner import (
    Budget,
    conservative_per_call_cost_ceiling,
    decide_for_strategy,
    execute_planned_routing_call,
)
from labs.lab05_model_routing.stats import compare_strategy_cost, compute_task_comparisons, summarize_all_strategies

LAB_ID = "lab05"

_EXPERIMENT_TO_STRATEGY = {
    "fixed_model_baseline": STRATEGY_ALWAYS_SMALL,
    "rule_based_routing": STRATEGY_ROUTER,
    "always_large": STRATEGY_ALWAYS_LARGE,
}
_ALL_EXPERIMENTS = [*_EXPERIMENT_TO_STRATEGY.keys(), "all", "backfill"]


def add_lab05_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Task/strategy scope (default: config lab05.default_profile).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan (including offline routing decisions), never "
        "call the API (always safe: ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab05 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab05-%Y%m%dT%H%M%SZ")


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


def _plan_for_experiment(experiment: str, profile: str) -> list:
    if experiment == "all":
        return build_plan_for_profile(profile)
    strategy = _EXPERIMENT_TO_STRATEGY[experiment]
    return build_routing_plan(task_ids_for(profile), [strategy], start_execution_index=1)


def _print_plan_header(small_model: str, large_model: str, strategies: list[str], task_ids: tuple, total_planned: int, live_enabled: bool) -> None:
    print("LAB05 -- Model Routing\n")
    print("Models:")
    print(f"  Small: {small_model}")
    print(f"  Large: {large_model}\n")
    print("Strategies:")
    for s in strategies:
        print(f"  {s}")
    print(f"\nTasks: {len(task_ids)}")
    print(f"Planned API Calls: {total_planned}\n")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_dry_run_preview(plan, small_model: str, large_model: str, pricing_by_model: dict, max_completion_tokens: int) -> float:
    print("\nDataset version: golden_dataset.json (13 tasks; see labs.lab05_model_routing.golden_dataset)")
    print("\nRouting Decisions (computed offline -- zero API calls):\n")
    estimated_max_cost = 0.0
    for call in plan:
        task = TASK_BY_ID[call.task_id]
        decision = decide_for_strategy(call.strategy, task, small_model, large_model)
        print(f"  [{call.execution_index}] task={call.task_id} strategy={call.strategy} -> {decision.selected_model} ({decision.routing_reason})")
        pricing = pricing_by_model.get(decision.selected_model)
        if pricing is not None:
            estimated_max_cost += conservative_per_call_cost_ceiling(pricing, max_completion_tokens, task.prompt)
    print(f"\nEstimated Maximum Cost (conservative, pre-flight): {_fmt_cost(estimated_max_cost)}")
    return estimated_max_cost


def _print_call_progress(m: RoutingMeasurement) -> None:
    print(f"\nTask: {m.task_id}")
    print(f"Category: {m.task_category}")
    print(f"Strategy: {m.strategy}\n")
    print(f"Selected Model: {m.selected_model}")
    print(f"Routing Reason: {m.routing_reason}")
    if not m.succeeded:
        print(f"\nFAILED: {m.error}")
        return
    print(f"\nPrompt Tokens:          {_fmt(m.prompt_tokens, 0)}")
    print(f"Completion Tokens:      {_fmt(m.completion_tokens, 0)}")
    print(f"Reasoning Tokens:       {_fmt(m.reasoning_tokens, 0)}")
    print(f"Non-Reasoning Tokens:   {_fmt(m.non_reasoning_completion_tokens, 0)}")
    print(f"Total Tokens:           {_fmt(m.total_tokens, 0)}")
    print(f"\nAnswer Correct:         {m.answer_correct}")
    print(f"Answer Complete:        {m.answer_complete}")
    print(f"Evaluation Status:      {m.evaluation_status}")
    print(f"Finish Reason:          {m.finish_reason}")
    print(f"\nClient Latency:         {_fmt(m.total_latency_ms)} ms")
    print(f"Estimated Cost:         {_fmt_cost(m.estimated_cost_usd)}")


def _print_strategy_comparison(summaries, cost_comparison) -> None:
    print("\n-- Model Routing Comparison --\n")
    for s in summaries:
        print(f"{s.strategy}:")
        print(f"  Tasks:                  {s.total_tasks}")
        print(f"  Correct Answers:        {s.correct_tasks}")
        print(f"  Accuracy:               {_fmt_pct(s.accuracy)} ({s.correct_tasks}/{s.total_tasks} tasks)")
        print(f"  Small Model Calls:      {s.small_model_calls}")
        print(f"  Large Model Calls:      {s.large_model_calls}")
        print(f"  Total Estimated Cost:   {_fmt_cost(s.total_estimated_cost_usd)}")
        print(f"  Mean Latency:           {_fmt(s.latency_ms.mean)} ms\n")

    if cost_comparison is not None:
        print(f"Cost Delta (routing - baseline):   {_fmt_cost(cost_comparison.absolute_cost_delta_usd)}")
        if cost_comparison.routing_cost_savings_percent is not None:
            if cost_comparison.routing_cost_savings_percent < 0:
                print(f"Overall Assessment:                Routing Cost Increase: +{_fmt(-cost_comparison.routing_cost_savings_percent)}%")
            else:
                print(f"Overall Assessment:                Routing Cost Savings: {_fmt(cost_comparison.routing_cost_savings_percent)}%")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab05_config = probe_ctx.config.lab05

    profile = args.profile or lab05_config.default_profile
    small_model = lab05_config.small_model
    large_model = lab05_config.large_model

    plan = _plan_for_experiment(args.experiment, profile)
    strategies = strategies_for_profile(profile) if args.experiment == "all" else [_EXPERIMENT_TO_STRATEGY[args.experiment]]
    task_ids = task_ids_for(profile)
    total_planned = len(plan)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(small_model, large_model, list(strategies), task_ids, total_planned, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab05_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        _print_dry_run_preview(plan, small_model, large_model, probe_ctx.config.pricing, lab05_config.max_completion_tokens)
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab05 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan (including routing decisions) without --live."
        )
        return

    if lab05_config.max_api_calls is not None and total_planned > lab05_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab05.max_api_calls ({lab05_config.max_api_calls}). No API calls were made."
        )
        return

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    ctx = build_context(requested_live=True, config_path=args.config)
    for model in (small_model, large_model):
        if ctx.config.pricing.get(model) is None:
            print(f"\nNo pricing configured for model '{model}': estimated_cost_usd will be None for its calls.")

    longest_prompt = max((t.prompt for t in TASK_BY_ID.values()), key=len)
    per_call_ceiling = max(
        (
            conservative_per_call_cost_ceiling(pricing, lab05_config.max_completion_tokens, longest_prompt)
            for pricing in (ctx.config.pricing.get(small_model), ctx.config.pricing.get(large_model))
            if pricing is not None
        ),
        default=0.0,
    )
    budget = Budget(
        max_api_calls=None,  # the pre-flight total_planned gate above already enforces this
        max_estimated_cost_usd=lab05_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )

    manifest = build_initial_manifest(experiment_run_id, lab05_config, profile, total_planned, ctx.config.pricing)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)

    measurements: list[RoutingMeasurement] = []
    stopped_reason = None

    for planned_call in plan:
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before call: {reason}")
            break

        task = TASK_BY_ID[planned_call.task_id]
        measurement = execute_planned_routing_call(ctx.client, ctx.config.retry, ctx.config, experiment_run_id, planned_call, task)
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        measurements.append(measurement)
        _print_call_progress(measurement)

    write_routing_measurements(run_dir, measurements)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    write_manifest(run_dir, manifest)

    summaries = summarize_all_strategies(list(strategies), measurements)
    write_strategy_summary_csv(run_dir, summaries)

    comparisons = (
        compute_task_comparisons(measurements) if STRATEGY_ALWAYS_SMALL in strategies and STRATEGY_ROUTER in strategies else []
    )
    write_task_comparison_csv(run_dir, comparisons)

    by_strategy = {s.strategy: s for s in summaries}
    cost_comparison = (
        compare_strategy_cost(by_strategy[STRATEGY_ALWAYS_SMALL], by_strategy[STRATEGY_ROUTER])
        if STRATEGY_ALWAYS_SMALL in by_strategy and STRATEGY_ROUTER in by_strategy
        else None
    )

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
            "strategies": [asdict(s) for s in summaries],
            "task_comparisons": [asdict(c) for c in comparisons],
            "measurements": [asdict(m) for m in measurements],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, measurements, summaries, comparisons, cost_comparison)
    write_analysis_report(run_dir, report_markdown)

    if successes:
        _print_strategy_comparison(summaries, cost_comparison)
        # Best-effort, strictly after every measurement/report file above is
        # already written -- a charting failure must never cost an
        # already-persisted measurement.
        try:
            from labs.lab05_model_routing.charts import generate_routing_charts

            chart_paths = generate_routing_charts(run_dir)
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
    parser = subparsers.add_parser(LAB_ID, help="Lab05 - Model Routing")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="fixed_model_baseline runs Always-Small only; rule_based_routing runs the router only; "
        "always_large runs Always-Large only (extended-profile baseline); all runs every strategy "
        "--profile enables (pilot=8, default=12, extended=18 calls); backfill (re)generates PNG "
        "charts from existing results, offline, without any API key.",
    )
    add_common_arguments(parser)
    add_lab05_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
