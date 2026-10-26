"""CLI wiring and orchestration for Lab01's `factorial` execution mode.

Invoked as `lab-cli lab01 factorial [...]` -- "factorial" is simply one more
value of the existing `experiment` positional argument lab01/cli.py already
has (alongside short_prompt/long_prompt/all), which is this project's
existing pattern for "which thing should this lab's CLI do", reused rather
than inventing a second sub-subcommand layer.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import build_context
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.factorial_matrix import (
    FactorialMatrixError,
    build_conditions,
    build_execution_plan,
)
from labs.lab01_token_anatomy.factorial_models import FactorialMeasurement, PlannedCall
from labs.lab01_token_anatomy.factorial_persistence import (
    ResumeConfigurationMismatch,
    append_measurement,
    build_initial_manifest,
    completed_pairs,
    factorial_run_dir,
    read_existing_measurements,
    read_manifest,
    validate_resume_fingerprint,
    write_analysis_report,
    write_condition_summary_csv,
    write_experiment_plan_csv,
    write_factorial_summary_json,
    write_manifest,
)
from labs.lab01_token_anatomy.factorial_report import build_analysis_report_markdown
from labs.lab01_token_anatomy.factorial_runner import (
    Budget,
    conservative_per_call_cost_ceiling,
    execute_planned_call,
)
from labs.lab01_token_anatomy.factorial_stats import summarize_all_conditions

SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high")


def add_factorial_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="factorial mode only: print and save the experiment plan, never call the API "
        "(always safe: ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="factorial mode only: restrict to these prompt types (default: all configured).",
    )
    parser.add_argument(
        "--reasoning-effort",
        nargs="+",
        default=None,
        choices=list(SUPPORTED_REASONING_EFFORTS),
        help="factorial mode only: restrict to these reasoning effort levels (default: all configured).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="factorial mode only: name this run, or resume an existing one with this id "
        "(default: an auto-generated timestamp-based id).",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("factorial-%Y%m%dT%H%M%SZ")


def _print_plan_header(model: str, factorial_config, conditions, plan, live_enabled: bool) -> None:
    print("LAB01 -- Full Factorial Experiment Plan\n")
    print(f"Model:                  {model}\n")
    print(f"Prompt Types:           {len({c.prompt_type for c in conditions})}")
    print(f"Reasoning Efforts:      {len({c.reasoning_effort for c in conditions})}")
    print(f"Temperature Levels:     {len({c.temperature for c in conditions})}")
    print(f"Top-p Levels:           {len({c.top_p for c in conditions})}\n")
    print(f"Unique Conditions:      {len(conditions)}")
    print(f"Repetitions:            {factorial_config.repetitions}\n")
    print(f"Planned API Calls:      {len(plan)}\n")
    print(f"Live API Execution:     {'Enabled' if live_enabled else 'Disabled'}")


def _print_call_progress(planned_call: PlannedCall, measurement: FactorialMeasurement, total_planned: int) -> None:
    condition = planned_call.condition
    print(f"\n[{planned_call.execution_index}/{total_planned}]")
    print(f"Prompt: {condition.prompt_type}")
    print(f"Effort: {condition.reasoning_effort}")
    print(f"Temperature: {condition.temperature}")
    print(f"Top-p: {condition.top_p}")
    print()
    if measurement.succeeded:
        reasoning = measurement.reasoning_tokens if measurement.reasoning_tokens is not None else "N/A"
        latency = f"{measurement.total_latency_ms:.1f} ms" if measurement.total_latency_ms is not None else "N/A"
        cost = f"${measurement.estimated_cost_usd:.8f}" if measurement.estimated_cost_usd is not None else "N/A"
        print(f"Completion Tokens: {measurement.completion_tokens}")
        print(f"Reasoning Tokens: {reasoning}")
        print(f"Latency: {latency}")
        print(f"Estimated Cost: {cost}")
    else:
        print(f"FAILED: {measurement.error}")
    print(f"\nProgress: {planned_call.execution_index} / {total_planned}")


def _print_completion_summary(measurements, total_planned: int, run_dir, stopped_reason) -> None:
    successes = [m for m in measurements if m.succeeded]
    failures = [m for m in measurements if not m.succeeded]
    total_tokens = sum(m.total_tokens for m in successes if m.total_tokens is not None)
    reasoning_tokens = sum(m.reasoning_tokens for m in successes if m.reasoning_tokens is not None)
    total_cost = sum(m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None)

    print(f"\n{'Experiment Completed' if stopped_reason is None else 'Experiment Stopped Early'}\n")
    if stopped_reason:
        print(f"Reason:               {stopped_reason}\n")
    print(f"Planned Calls:        {total_planned}")
    print(f"Successful Calls:     {len(successes)}")
    print(f"Failed Calls:         {len(failures)}\n")
    print(f"Total Tokens:         {total_tokens}")
    print(f"Reasoning Tokens:     {reasoning_tokens}")
    print(f"Estimated Cost:       ${total_cost:.6f}\n")
    print(f"Results Directory:    {run_dir}")


def run_factorial(args: argparse.Namespace) -> None:
    # --dry-run always wins over --live: a plan preview must never need an
    # API key or touch the network, no matter what else was passed.
    requested_live = bool(args.live) and not args.dry_run
    ctx = build_context(requested_live=requested_live, config_path=args.config)

    factorial_config = ctx.config.lab01_factorial
    if factorial_config is None:
        print(
            "Lab01 factorial mode requires a [lab01.factorial] section in your config file "
            "-- see config/default.toml for a documented example to add to your own config "
            "(e.g. config/live.toml)."
        )
        return
    if not factorial_config.enabled:
        print("Lab01 factorial mode is disabled for this config (lab01.factorial.enabled = false).")
        return

    model = args.model or ctx.config.groq.default_model
    prompts = args.prompt or list(factorial_config.prompts)
    reasoning_efforts = args.reasoning_effort or list(factorial_config.reasoning_efforts)
    repetitions = args.repetitions or factorial_config.repetitions

    unknown_prompts = [p for p in prompts if p not in factorial_config.prompts]
    if unknown_prompts:
        print(
            f"--prompt value(s) {unknown_prompts} are not in this config's "
            f"lab01.factorial.prompts {list(factorial_config.prompts)}; nothing executed."
        )
        return
    unknown_efforts = [e for e in reasoning_efforts if e not in factorial_config.reasoning_efforts]
    if unknown_efforts:
        print(
            f"--reasoning-effort value(s) {unknown_efforts} are not in this config's "
            f"lab01.factorial.reasoning_efforts {list(factorial_config.reasoning_efforts)}; nothing executed."
        )
        return

    try:
        conditions = build_conditions(
            prompts, reasoning_efforts, factorial_config.temperatures, factorial_config.top_p_values
        )
    except FactorialMatrixError as exc:
        print(f"Invalid factorial configuration: {exc}")
        return

    plan = build_execution_plan(
        conditions,
        repetitions,
        factorial_config.randomize_execution_order,
        factorial_config.randomization_seed,
        factorial_config.use_paired_api_seed,
    )

    _print_plan_header(model, factorial_config, conditions, plan, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = factorial_run_dir(ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not ctx.live:
        print(
            "\nLab01 factorial mode requires --live (and execution.live_api_enabled=true) to "
            "make real measurements against the Groq API; nothing was executed. Use --dry-run "
            "to preview the plan without --live."
        )
        return

    try:
        validate_resume_fingerprint(run_dir, factorial_config, model)
    except ResumeConfigurationMismatch as exc:
        print(f"\n{exc}")
        return

    existing_measurements = read_existing_measurements(run_dir)
    already_done = completed_pairs(existing_measurements)
    remaining_plan = [c for c in plan if (c.condition.condition_id, c.repetition_index) not in already_done]

    manifest = read_manifest(run_dir)
    if manifest is None:
        manifest = build_initial_manifest(experiment_run_id, factorial_config, model, len(conditions), len(plan))
        write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)

    if already_done:
        print(
            f"\nResuming experiment_run_id '{experiment_run_id}': {len(already_done)} call(s) "
            f"already completed, {len(remaining_plan)} remaining."
        )
    elif not remaining_plan:
        print(f"\nAll {len(plan)} planned calls were already completed; nothing new to execute.")

    pricing = ctx.config.pricing.get(model)
    if pricing is None:
        print(
            f"\nNo pricing configured for model '{model}': estimated_cost_usd will be None "
            "for every call, and the cost budget cannot be enforced (only max_api_calls will be)."
        )

    longest_prompt = max(EXPERIMENTS.values(), key=len)
    per_call_ceiling = (
        conservative_per_call_cost_ceiling(pricing, factorial_config.max_completion_tokens, longest_prompt)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=factorial_config.max_api_calls,
        max_estimated_cost_usd=factorial_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )
    # Calls already completed in a prior partial run count toward the budget
    # too, so resuming can never push total spend/calls past the cap.
    for prior in existing_measurements:
        budget.record_call(prior.estimated_cost_usd if prior.succeeded else None)

    stopped_reason = None
    measurements: list[FactorialMeasurement] = list(existing_measurements)
    for position, planned_call in enumerate(remaining_plan, start=1):
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before call {position}/{len(remaining_plan)}: {reason}")
            break

        measurement = execute_planned_call(
            ctx.client,
            ctx.config.retry,
            model,
            pricing,
            factorial_config.max_completion_tokens,
            experiment_run_id,
            planned_call,
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        append_measurement(run_dir, measurement, is_first_write=(len(measurements) == 0))
        measurements.append(measurement)

        _print_call_progress(planned_call, measurement, len(plan))

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    write_manifest(run_dir, manifest)

    planned_calls_by_condition = {c.condition_id: repetitions for c in conditions}
    summaries = summarize_all_conditions(conditions, planned_calls_by_condition, measurements)
    write_condition_summary_csv(run_dir, summaries)

    write_factorial_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "planned_calls": len(plan),
            "successful_calls": sum(1 for m in measurements if m.succeeded),
            "failed_calls": sum(1 for m in measurements if not m.succeeded),
            "stopped_reason": stopped_reason,
            "total_estimated_cost_usd": budget.accumulated_estimated_cost_usd,
            "conditions": [asdict(s) for s in summaries],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, conditions, measurements, summaries)
    write_analysis_report(run_dir, report_markdown)

    _print_completion_summary(measurements, len(plan), run_dir, stopped_reason)
