"""CLI wiring and orchestration for Lab02 -- Context Growth & Context Bloat.

Execution modes (the `experiment` positional argument):
  - minimal_context / expanded_context / irrelevant_context: run just that
    one context condition, `--repetitions` times.
  - multi_turn_growth: run just the growing multi-turn conversation,
    `--turns` turns.
  - all: all three context conditions (`--repetitions` times each) THEN
    the multi-turn conversation -- this project's documented 10-call
    default (3 conditions x 2 repetitions + 4 turns).
  - backfill: offline, read-only chart (re)generation from existing
    results -- see labs.lab02_context_growth.charts_cli.

Every live-executing mode writes into its own
`results/lab02_context_growth/<experiment_run_id>/` directory (see
labs.lab02_context_growth.persistence) and never touches the pre-existing
flat `results/lab02_context_growth.json`/`.csv` files from before this
redesign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import add_common_arguments, build_context
from labs.lab02_context_growth.charts_cli import add_backfill_arguments, run_backfill
from labs.lab02_context_growth.experiments import CONDITIONS, CONTEXT_CONDITION_ORDER
from labs.lab02_context_growth.matrix import build_context_plan
from labs.lab02_context_growth.models import ContextMeasurement, TurnMeasurement
from labs.lab02_context_growth.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab02_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_condition_summary_csv,
    write_context_measurements,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
    write_multi_turn_measurements,
)
from labs.lab02_context_growth.report import build_analysis_report_markdown
from labs.lab02_context_growth.runner import (
    Budget,
    conservative_per_call_cost_ceiling,
    execute_multi_turn_growth,
    execute_planned_context_call,
)
from labs.lab02_context_growth.stats import summarize_all_context_conditions

LAB_ID = "lab02"

_ALL_EXPERIMENTS = [*CONTEXT_CONDITION_ORDER, "multi_turn_growth", "all", "backfill"]


def add_lab02_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Groq model id (default: config groq.default_model).")
    parser.add_argument(
        "--turns", type=int, default=None, help="Override the number of turns for multi_turn_growth."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan, never call the API (always safe: "
        "ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab02 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab02-%Y%m%dT%H%M%SZ")


def _conditions_for(experiment: str) -> list[str]:
    if experiment == "all":
        return list(CONTEXT_CONDITION_ORDER)
    if experiment in CONTEXT_CONDITION_ORDER:
        return [experiment]
    return []


def _print_plan_header(model: str, conditions: list[str], repetitions: int, run_multi_turn: bool, turns: int, total_planned: int, live_enabled: bool) -> None:
    print("LAB02 -- Context Growth & Context Bloat\n")
    print(f"Model: {model}\n")
    print("Conditions:")
    for condition_id in conditions:
        print(f"  {condition_id}")
    if run_multi_turn:
        print("  multi_turn_growth")
    print()
    print(f"Repetitions: {repetitions}")
    print(f"Multi-turn Turns: {turns if run_multi_turn else 0}\n")
    print(f"Planned API Calls: {total_planned}\n")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _print_context_call_progress(measurement: ContextMeasurement, repetitions: int) -> None:
    print(f"\nCondition: {measurement.condition_id}")
    print(f"Run: {measurement.repetition_index}/{repetitions}\n")
    if not measurement.succeeded:
        print(f"FAILED: {measurement.error}")
        return
    print(f"Prompt Tokens:         {_fmt(measurement.prompt_tokens, 0)}")
    print(f"Completion Tokens:     {_fmt(measurement.completion_tokens, 0)}")
    print(f"Reasoning Tokens:      {_fmt(measurement.reasoning_tokens, 0)}")
    print(f"Non-Reasoning Tokens:  {_fmt(measurement.non_reasoning_completion_tokens, 0)}")
    print(f"Total Tokens:          {_fmt(measurement.total_tokens, 0)}")
    print(f"Latency:               {_fmt(measurement.total_latency_ms)} ms")
    print(f"Estimated Cost:        {_fmt_cost(measurement.estimated_cost_usd)}")
    print(f"Finish Reason:         {measurement.finish_reason}")
    print(f"Answer Correct:        {measurement.answer_correct} ({measurement.answer_evaluation_status})")


def _print_condition_summary(condition_id: str, measurements: list[ContextMeasurement]) -> None:
    from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats

    successes = [m for m in measurements if m.succeeded]
    correct = sum(1 for m in successes if m.answer_correct is True)
    prompt_stats = compute_descriptive_stats([m.prompt_tokens for m in successes])
    completion_stats = compute_descriptive_stats([m.completion_tokens for m in successes])
    reasoning_stats = compute_descriptive_stats([m.reasoning_tokens for m in successes])
    latency_stats = compute_descriptive_stats([m.total_latency_ms for m in successes])
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]

    print(f"\n-- {condition_id} summary --\n")
    print(f"Successful Calls:       {len(successes)}/{len(measurements)}")
    print(f"Mean Prompt Tokens:     {_fmt(prompt_stats.mean)}")
    print(f"Mean Completion Tokens: {_fmt(completion_stats.mean)}")
    print(f"Mean Reasoning Tokens:  {_fmt(reasoning_stats.mean)}")
    print(f"Mean Latency:           {_fmt(latency_stats.mean)} ms")
    print(f"Mean Estimated Cost:    {_fmt_cost((sum(costs) / len(costs)) if costs else None)}")
    print(f"Correct Answers:        {correct}/{len(successes)}")


def _print_turn_progress(measurement: TurnMeasurement, total_turns: int) -> None:
    print(f"\nTurn: {measurement.turn_index}/{total_turns}\n")
    if not measurement.succeeded:
        print(f"FAILED: {measurement.error}")
        return
    print(f"Messages Count:            {measurement.messages_count}")
    print(f"Prompt Tokens:             {_fmt(measurement.prompt_tokens, 0)}")
    print(f"Incremental Prompt Tokens: {_fmt(measurement.incremental_prompt_tokens, 0)}")
    print(f"Cumulative Prompt Tokens:  {_fmt(measurement.cumulative_prompt_tokens, 0)}")
    print(f"Latency:                   {_fmt(measurement.total_latency_ms)} ms")
    print(f"Cumulative Estimated Cost: {_fmt_cost(measurement.cumulative_estimated_cost_usd)}")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab02_config = probe_ctx.config.lab02
    if not lab02_config.enabled:
        print("Lab02 is disabled for this config (lab02.enabled = false).")
        return

    model = args.model or probe_ctx.config.groq.default_model
    repetitions = args.repetitions or lab02_config.repetitions
    turns = args.turns if args.turns is not None else lab02_config.multi_turn_turns

    conditions = _conditions_for(args.experiment)
    run_multi_turn = args.experiment in ("all", "multi_turn_growth")

    context_plan = build_context_plan(
        conditions, repetitions, lab02_config.randomize_context_order, lab02_config.randomization_seed
    )
    total_planned = len(context_plan) + (turns if run_multi_turn else 0)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, conditions, repetitions, run_multi_turn, turns, total_planned, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab02_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, context_plan, turns if run_multi_turn else 0)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab02 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab02_config.max_api_calls is not None and total_planned > lab02_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab02.max_api_calls ({lab02_config.max_api_calls}). Lower --repetitions/--turns, "
            "or raise lab02.max_api_calls deliberately in your config if you intend more. "
            "No API calls were made."
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

    longest_context = max(CONDITIONS.values(), key=len)
    per_call_ceiling = (
        conservative_per_call_cost_ceiling(pricing, lab02_config.max_completion_tokens, longest_context)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=None,  # the pre-flight total_planned gate above already enforces this
        max_estimated_cost_usd=lab02_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )

    manifest = build_initial_manifest(experiment_run_id, lab02_config, model, len(context_plan), turns if run_multi_turn else 0)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, context_plan, turns if run_multi_turn else 0)

    context_measurements: list[ContextMeasurement] = []
    measurements_by_condition: dict[str, list[ContextMeasurement]] = {}
    completed_by_condition: dict[str, int] = {}
    stopped_reason = None

    for planned_call in context_plan:
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before context call: {reason}")
            break

        measurement = execute_planned_context_call(
            ctx.client, ctx.config.retry, model, pricing, lab02_config.max_completion_tokens, experiment_run_id, planned_call
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        context_measurements.append(measurement)
        measurements_by_condition.setdefault(measurement.condition_id, []).append(measurement)
        _print_context_call_progress(measurement, repetitions)

        completed_by_condition[measurement.condition_id] = completed_by_condition.get(measurement.condition_id, 0) + 1
        if completed_by_condition[measurement.condition_id] == repetitions:
            _print_condition_summary(measurement.condition_id, measurements_by_condition[measurement.condition_id])

    write_context_measurements(run_dir, context_measurements)

    turn_measurements: list[TurnMeasurement] = []
    if run_multi_turn and stopped_reason is None:
        turn_measurements = execute_multi_turn_growth(
            ctx.client, ctx.config.retry, model, pricing, lab02_config.max_completion_tokens, experiment_run_id, turns, budget=budget
        )
        for t in turn_measurements:
            _print_turn_progress(t, turns)
        if len(turn_measurements) < turns:
            stopped_reason = stopped_reason or "budget exhausted during multi_turn_growth"
    write_multi_turn_measurements(run_dir, turn_measurements)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    write_manifest(run_dir, manifest)

    planned_calls_by_condition = {c: repetitions for c in conditions}
    summaries = summarize_all_context_conditions(conditions, planned_calls_by_condition, context_measurements)
    write_condition_summary_csv(run_dir, summaries)

    successes = [m for m in context_measurements if m.succeeded]
    turn_successes = [t for t in turn_measurements if t.succeeded]
    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "planned_calls": total_planned,
            "successful_context_calls": len(successes),
            "failed_context_calls": len(context_measurements) - len(successes),
            "successful_turn_calls": len(turn_successes),
            "failed_turn_calls": len(turn_measurements) - len(turn_successes),
            "stopped_reason": stopped_reason,
            "total_estimated_cost_usd": budget.accumulated_estimated_cost_usd,
            "conditions": [asdict(s) for s in summaries],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, context_measurements, turn_measurements, summaries)
    write_analysis_report(run_dir, report_markdown)

    if successes or turn_successes:
        # Best-effort, strictly after every measurement/report file above is
        # already written -- a charting failure must never cost an
        # already-persisted measurement.
        try:
            from labs.lab02_context_growth.charts import generate_context_growth_charts

            chart_paths = generate_context_growth_charts(run_dir)
            if chart_paths:
                print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\n{'Experiment Completed' if stopped_reason is None else 'Experiment Stopped Early'}\n")
    if stopped_reason:
        print(f"Reason: {stopped_reason}\n")
    print(f"Planned Calls:     {total_planned}")
    print(f"Successful Calls:  {len(successes) + len(turn_successes)}")
    print(f"Results Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab02 - Context Growth & Context Bloat")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="minimal_context/expanded_context/irrelevant_context run one context condition; "
        "multi_turn_growth runs the growing conversation; all runs every context condition "
        "then multi_turn_growth (this project's documented 10-call default); backfill "
        "(re)generates PNG charts from existing results, offline, without any API key.",
    )
    add_common_arguments(parser)
    add_lab02_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
