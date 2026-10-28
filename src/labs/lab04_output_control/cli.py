"""CLI wiring and orchestration for Lab04 -- Output Control & Invocation
Optimization.

Execution modes (the `experiment` positional argument):
  - output_limit_sweep: the retry-policy-extraction task sent at each of
    `lab04.output_limit_values`, `lab04.output_limit_repetitions` times
    each (default: 3 values x 2 repetitions = 6 calls).
  - invocation_comparison: the same three trivia questions answered once
    combined (1 call) and once per-question (3 calls) -- always exactly 4
    calls, not configurable in count.
  - all: both of the above -- this project's documented 10-call default.
  - backfill: offline, read-only chart (re)generation from existing
    results -- see labs.lab04_output_control.charts_cli.

Every live-executing mode writes into its own
`results/lab04_output_control/<experiment_run_id>/` directory and never
touches the pre-existing flat `results/lab04_output_control.json`/`.csv`
files from before this redesign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import add_common_arguments, build_context
from labs.lab04_output_control.charts_cli import add_backfill_arguments, run_backfill
from labs.lab04_output_control.evaluation import RETRY_POLICY_SYSTEM_MESSAGE, RETRY_POLICY_USER_MESSAGE
from labs.lab04_output_control.experiments import build_invocation_single_messages, condition_label_for
from labs.lab04_output_control.matrix import build_invocation_plan, build_output_limit_plan
from labs.lab04_output_control.models import InvocationMeasurement, OutputLimitMeasurement
from labs.lab04_output_control.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab04_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_condition_summary_csv,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_invocation_comparison_csv,
    write_invocation_measurements,
    write_manifest,
    write_output_limit_measurements,
)
from labs.lab04_output_control.report import build_analysis_report_markdown
from labs.lab04_output_control.runner import (
    Budget,
    conservative_per_call_cost_ceiling,
    execute_planned_output_limit_call,
    run_invocation_strategy,
)
from labs.lab04_output_control.stats import (
    compare_invocation_strategies,
    summarize_all_output_limit_conditions,
    summarize_invocation_strategy,
)

LAB_ID = "lab04"

_ALL_EXPERIMENTS = ["output_limit_sweep", "invocation_comparison", "all", "backfill"]

# Used only as a conservative, pre-flight cost-ceiling estimate (never an
# actual request) -- the longer of this lab's two static prompts.
_LONGEST_STATIC_PROMPT = max(
    RETRY_POLICY_SYSTEM_MESSAGE + RETRY_POLICY_USER_MESSAGE,
    build_invocation_single_messages()[0]["content"],
    key=len,
)


def add_lab04_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Must be reasoning-capable (see config lab04.model).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan, never call the API (always safe: "
        "ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab04 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab04-%Y%m%dT%H%M%SZ")


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


def _print_plan_header(
    model: str, run_output_limit: bool, run_invocation: bool, output_limit_repetitions: int, total_planned: int, live_enabled: bool
) -> None:
    print("LAB04 -- Output Control & Invocation Optimization\n")
    print(f"Model: {model}\n")
    print("Experiments:")
    if run_output_limit:
        print("  output_limit_sweep")
    if run_invocation:
        print("  invocation_comparison")
    print()
    if run_output_limit:
        print(f"Output-Limit Repetitions (per value): {output_limit_repetitions}")
    print(f"Planned API Calls: {total_planned}\n")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_output_limit_progress(m: OutputLimitMeasurement, repetitions: int) -> None:
    print(f"\nExperiment: output_limit_sweep")
    print(f"Condition: {m.condition_id}")
    print(f"Run: {m.repetition_index}/{repetitions}\n")
    if not m.succeeded:
        print(f"FAILED: {m.error}")
        return
    print(f"Max Completion Tokens:      {m.max_completion_tokens}")
    print(f"Prompt Tokens:              {_fmt(m.prompt_tokens, 0)}")
    print(f"Completion Tokens:          {_fmt(m.completion_tokens, 0)}")
    print(f"Reasoning Tokens:           {_fmt(m.reasoning_tokens, 0)}")
    print(f"Non-Reasoning Tokens:       {_fmt(m.non_reasoning_completion_tokens, 0)}")
    print(f"Total Tokens:               {_fmt(m.total_tokens, 0)}")
    print(f"Finish Reason:              {m.finish_reason}")
    print(f"Output Truncated:           {m.output_truncated}")
    print(f"Required Fields:            {m.required_field_count}")
    print(f"Present Fields:             {m.present_field_count}")
    print(f"Correct Fields:             {m.correct_field_count}")
    print(f"Answer Completeness:        {_fmt_pct(m.field_completeness_ratio)}")
    print(f"Answer Correct (all fields):{m.overall_answer_accepted}")
    print(f"Client Latency:             {_fmt(m.total_latency_ms)} ms")
    print(f"Estimated Cost:             {_fmt_cost(m.estimated_cost_usd)}")


def _print_output_limit_condition_summary(condition_id: str, measurements: list[OutputLimitMeasurement]) -> None:
    from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats

    successes = [m for m in measurements if m.succeeded]
    accepted = sum(1 for m in successes if m.overall_answer_accepted)
    truncated = sum(1 for m in successes if m.output_truncated is True)
    completion_stats = compute_descriptive_stats([m.completion_tokens for m in successes])
    reasoning_stats = compute_descriptive_stats([m.reasoning_tokens for m in successes])
    latency_stats = compute_descriptive_stats([m.total_latency_ms for m in successes])
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]

    print(f"\n-- {condition_id} summary --\n")
    print(f"Successful Calls:       {len(successes)}/{len(measurements)}")
    print(f"Mean Completion Tokens: {_fmt(completion_stats.mean)}")
    print(f"Mean Reasoning Tokens:  {_fmt(reasoning_stats.mean)}")
    print(f"Mean Latency:           {_fmt(latency_stats.mean)} ms")
    print(f"Mean Estimated Cost:    {_fmt_cost((sum(costs) / len(costs)) if costs else None)}")
    print(f"Accepted Answers:       {accepted}/{len(successes)}")
    print(f"Truncated Responses:    {truncated}/{len(successes)}")


def _print_invocation_progress(m: InvocationMeasurement) -> None:
    label = "Single" if m.strategy == "single" else f"Multiple (question {m.question_index})"
    print(f"\nExperiment: invocation_comparison -- {label}\n")
    if not m.succeeded:
        print(f"FAILED: {m.error}")
        return
    print(f"Prompt Tokens:         {_fmt(m.prompt_tokens, 0)}")
    print(f"Completion Tokens:     {_fmt(m.completion_tokens, 0)}")
    print(f"Finish Reason:         {m.finish_reason}")
    print(f"Answer Accepted:       {m.overall_answer_accepted}")
    print(f"Client Latency:        {_fmt(m.client_elapsed_ms)} ms")
    print(f"Estimated Cost:        {_fmt_cost(m.estimated_cost_usd)}")


def _print_invocation_comparison_summary(comparison) -> None:
    single, multiple = comparison.single, comparison.multiple
    print("\n-- invocation_comparison summary --\n")
    print("Single Invocation:")
    print(f"  API Calls:               {single.api_call_count}")
    print(f"  Total Prompt Tokens:     {single.total_prompt_tokens}")
    print(f"  Total Completion Tokens: {single.total_completion_tokens}")
    print(f"  Total Estimated Cost:    {_fmt_cost(single.total_estimated_cost_usd)}")
    print(f"  Workflow Wall-Clock:     {_fmt(single.workflow_wall_clock_ms)} ms")
    print(f"  Correct Answers:         {single.correct_answers}/{single.api_call_count}")
    print("\nMultiple Invocations:")
    print(f"  API Calls:               {multiple.api_call_count}")
    print(f"  Total Prompt Tokens:     {multiple.total_prompt_tokens}")
    print(f"  Total Completion Tokens: {multiple.total_completion_tokens}")
    print(f"  Total Estimated Cost:    {_fmt_cost(multiple.total_estimated_cost_usd)}")
    print(f"  Workflow Wall-Clock:     {_fmt(multiple.workflow_wall_clock_ms)} ms")
    print(f"  Correct Answers:         {multiple.correct_answers}/{multiple.api_call_count}")
    print(f"\nCost Difference:         {_fmt_cost(comparison.absolute_cost_difference_usd)} ({_fmt(comparison.relative_cost_difference_percent)}%)")
    print(f"Latency Difference:      {_fmt(comparison.workflow_latency_difference_ms)} ms")
    print(f"Quality Comparison:      {comparison.quality_comparison}")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab04_config = probe_ctx.config.lab04
    if not lab04_config.enabled:
        print("Lab04 is disabled for this config (lab04.enabled = false).")
        return

    model = args.model or lab04_config.model
    output_limit_repetitions = args.repetitions or lab04_config.output_limit_repetitions

    run_output_limit = args.experiment in ("output_limit_sweep", "all")
    run_invocation = args.experiment in ("invocation_comparison", "all")

    output_limit_plan = (
        build_output_limit_plan(lab04_config.output_limit_values, output_limit_repetitions, start_execution_index=1)
        if run_output_limit
        else []
    )
    invocation_plan = (
        build_invocation_plan(start_execution_index=len(output_limit_plan) + 1) if run_invocation else []
    )
    total_planned = len(output_limit_plan) + len(invocation_plan)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, run_output_limit, run_invocation, output_limit_repetitions, total_planned, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab04_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, output_limit_plan, invocation_plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab04 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab04_config.max_api_calls is not None and total_planned > lab04_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab04.max_api_calls ({lab04_config.max_api_calls}). Lower --repetitions, or raise "
            "lab04.max_api_calls deliberately in your config if you intend more. No API calls were made."
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

    largest_budget = max(
        [*lab04_config.output_limit_values, lab04_config.invocation_max_completion_tokens]
    )
    per_call_ceiling = (
        conservative_per_call_cost_ceiling(pricing, largest_budget, _LONGEST_STATIC_PROMPT)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=None,  # the pre-flight total_planned gate above already enforces this
        max_estimated_cost_usd=lab04_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )

    manifest = build_initial_manifest(experiment_run_id, lab04_config, model, total_planned)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, output_limit_plan, invocation_plan)

    output_limit_measurements: list[OutputLimitMeasurement] = []
    measurements_by_condition: dict[str, list[OutputLimitMeasurement]] = {}
    completed_by_condition: dict[str, int] = {}
    stopped_reason = None

    for planned_call in output_limit_plan:
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before call: {reason}")
            break

        measurement = execute_planned_output_limit_call(
            ctx.client, ctx.config.retry, model, pricing, lab04_config.reasoning_effort, experiment_run_id, planned_call
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        output_limit_measurements.append(measurement)
        measurements_by_condition.setdefault(measurement.condition_id, []).append(measurement)
        _print_output_limit_progress(measurement, output_limit_repetitions)

        completed_by_condition[measurement.condition_id] = completed_by_condition.get(measurement.condition_id, 0) + 1
        if completed_by_condition[measurement.condition_id] == output_limit_repetitions:
            _print_output_limit_condition_summary(measurement.condition_id, measurements_by_condition[measurement.condition_id])

    write_output_limit_measurements(run_dir, output_limit_measurements)

    invocation_measurements: list[InvocationMeasurement] = []
    invocation_comparison = None
    if run_invocation and stopped_reason is None:
        single_calls = [c for c in invocation_plan if c.strategy == "single"]
        multiple_calls = [c for c in invocation_plan if c.strategy == "multiple"]

        single_result = run_invocation_strategy(
            ctx.client, ctx.config.retry, model, pricing, lab04_config.invocation_max_completion_tokens,
            lab04_config.reasoning_effort, experiment_run_id, single_calls, budget,
        )
        for m in single_result.measurements:
            _print_invocation_progress(m)

        multiple_result = run_invocation_strategy(
            ctx.client, ctx.config.retry, model, pricing, lab04_config.invocation_max_completion_tokens,
            lab04_config.reasoning_effort, experiment_run_id, multiple_calls, budget,
        )
        for m in multiple_result.measurements:
            _print_invocation_progress(m)

        invocation_measurements = single_result.measurements + multiple_result.measurements
        if len(invocation_measurements) < len(invocation_plan):
            stopped_reason = stopped_reason or "budget exhausted during invocation_comparison"

        single_summary = summarize_invocation_strategy("single", single_result.measurements, single_result.workflow_wall_clock_ms)
        multiple_summary = summarize_invocation_strategy("multiple", multiple_result.measurements, multiple_result.workflow_wall_clock_ms)
        invocation_comparison = compare_invocation_strategies(single_summary, multiple_summary)
        _print_invocation_comparison_summary(invocation_comparison)

    write_invocation_measurements(run_dir, invocation_measurements)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    manifest["experiment_mode"] = args.experiment
    write_manifest(run_dir, manifest)

    conditions_list = [(condition_label_for(v), v) for v in lab04_config.output_limit_values] if run_output_limit else []
    planned_calls_by_condition = {cid: output_limit_repetitions for cid, _ in conditions_list}
    output_limit_summaries = summarize_all_output_limit_conditions(conditions_list, planned_calls_by_condition, output_limit_measurements)
    write_condition_summary_csv(run_dir, output_limit_summaries)
    if invocation_comparison is not None:
        write_invocation_comparison_csv(run_dir, invocation_comparison)

    ol_successes = [m for m in output_limit_measurements if m.succeeded]
    inv_successes = [m for m in invocation_measurements if m.succeeded]
    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "experiment_mode": args.experiment,
            "planned_calls": total_planned,
            "successful_output_limit_calls": len(ol_successes),
            "failed_output_limit_calls": len(output_limit_measurements) - len(ol_successes),
            "successful_invocation_calls": len(inv_successes),
            "failed_invocation_calls": len(invocation_measurements) - len(inv_successes),
            "stopped_reason": stopped_reason,
            "total_estimated_cost_usd": budget.accumulated_estimated_cost_usd,
            "output_limit_conditions": [asdict(s) for s in output_limit_summaries],
            "output_limit_measurements": [asdict(m) for m in output_limit_measurements],
            "invocation_comparison": (
                {
                    "single": asdict(invocation_comparison.single),
                    "multiple": asdict(invocation_comparison.multiple),
                    "absolute_cost_difference_usd": invocation_comparison.absolute_cost_difference_usd,
                    "relative_cost_difference_percent": invocation_comparison.relative_cost_difference_percent,
                    "workflow_latency_difference_ms": invocation_comparison.workflow_latency_difference_ms,
                    "quality_comparison": invocation_comparison.quality_comparison,
                }
                if invocation_comparison is not None
                else None
            ),
            "invocation_measurements": [asdict(m) for m in invocation_measurements],
        },
    )

    report_markdown = build_analysis_report_markdown(
        manifest, output_limit_measurements, invocation_measurements, output_limit_summaries, invocation_comparison
    )
    write_analysis_report(run_dir, report_markdown)

    if ol_successes or inv_successes:
        # Best-effort, strictly after every measurement/report file above is
        # already written -- a charting failure must never cost an
        # already-persisted measurement.
        try:
            from labs.lab04_output_control.charts import generate_output_control_charts

            chart_paths = generate_output_control_charts(run_dir)
            if chart_paths:
                print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
        except Exception as exc:  # noqa: BLE001 -- see comment above
            print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\n{'Experiment Completed' if stopped_reason is None else 'Experiment Stopped Early'}\n")
    if stopped_reason:
        print(f"Reason: {stopped_reason}\n")
    print(f"Planned Calls:     {total_planned}")
    print(f"Successful Calls:  {len(ol_successes) + len(inv_successes)}")
    print(f"Results Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab04 - Output Control & Invocation Optimization")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="output_limit_sweep runs the retry-policy-extraction task at each configured "
        "token budget; invocation_comparison runs the fixed single-vs-multiple-call "
        "comparison (always 4 calls); all runs both (this project's documented 10-call "
        "default); backfill (re)generates PNG charts from existing results, offline, "
        "without any API key.",
    )
    add_common_arguments(parser)
    add_lab04_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
