"""CLI wiring and orchestration for Lab01's `effort` execution mode.

Invoked as `lab-cli lab01 effort [...]` -- "effort" is one more value of the
existing `experiment` positional argument lab01/cli.py already has
(alongside short_prompt/long_prompt/all), the same pattern the retired
`factorial` mode used.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone

from lab_core.lab_cli_support import build_context
from labs.lab01_token_anatomy.experiments import EXPERIMENTS
from labs.lab01_token_anatomy.effort_matrix import EffortMatrixError, build_conditions, build_execution_plan
from labs.lab01_token_anatomy.effort_models import EffortMeasurement, PlannedEffortCall
from labs.lab01_token_anatomy.effort_persistence import (
    ResumeConfigurationMismatch,
    append_measurement,
    build_initial_manifest,
    completed_pairs,
    effort_run_dir,
    read_existing_measurements,
    read_manifest,
    validate_resume_fingerprint,
    write_analysis_report,
    write_condition_summary_csv,
    write_effort_summary_json,
    write_experiment_plan_csv,
    write_manifest,
)
from labs.lab01_token_anatomy.effort_report import build_analysis_report_markdown
from labs.lab01_token_anatomy.effort_runner import (
    Budget,
    conservative_per_call_cost_ceiling,
    execute_planned_call,
)
from labs.lab01_token_anatomy.effort_stats import summarize_all_effort_conditions

SUPPORTED_REASONING_EFFORTS = ("low", "medium", "high")


def add_effort_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="effort/factorial modes only: print and save the experiment plan, never call "
        "the API (always safe: ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--prompt",
        nargs="+",
        default=None,
        choices=list(EXPERIMENTS.keys()),
        help="effort mode only: restrict to these prompt types (default: all configured).",
    )
    parser.add_argument(
        "--reasoning-effort",
        nargs="+",
        default=None,
        choices=list(SUPPORTED_REASONING_EFFORTS),
        help="effort mode only: restrict to these reasoning effort levels (default: all configured).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="effort mode only: name this run, or resume an existing one with this id "
        "(default: an auto-generated timestamp-based id).",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("effort-%Y%m%dT%H%M%SZ")


def _print_plan_header(model: str, conditions, plan, live_enabled: bool) -> None:
    prompts = sorted({c.prompt_type for c in conditions})
    efforts = [e for e in SUPPORTED_REASONING_EFFORTS if e in {c.reasoning_effort for c in conditions}]
    repetitions = len(plan) // len(conditions) if conditions else 0

    print("LAB01 -- Reasoning Effort Comparison\n")
    print(f"Model: {model}\n")
    print("Prompts:")
    for prompt_type in prompts:
        print(f"  {prompt_type}")
    print()
    print("Reasoning Efforts:")
    for effort in efforts:
        print(f"  {effort}")
    print()
    print(f"Repetitions: {repetitions}\n")
    print(f"Unique Conditions: {len(conditions)}")
    print(f"Planned API Calls: {len(plan)}\n")
    print("Temperature: provider default (not explicitly set)")
    print("Top-p: provider default (not explicitly set)")
    print(f"\nLive API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_call_progress(planned_call: PlannedEffortCall, measurement: EffortMeasurement, repetitions: int) -> None:
    condition = planned_call.condition
    print(f"\nEffort: {condition.reasoning_effort} (prompt: {condition.prompt_type})")
    print(f"\nRun: {planned_call.repetition_index}/{repetitions}\n")
    if measurement.succeeded:
        reasoning = measurement.reasoning_tokens if measurement.reasoning_tokens is not None else "N/A"
        non_reasoning = (
            measurement.non_reasoning_completion_tokens
            if measurement.non_reasoning_completion_tokens is not None
            else "N/A"
        )
        latency = f"{measurement.total_latency_ms:.1f} ms" if measurement.total_latency_ms is not None else "N/A"
        cost = f"${measurement.estimated_cost_usd:.8f}" if measurement.estimated_cost_usd is not None else "N/A"
        print(f"Prompt Tokens:         {measurement.prompt_tokens}")
        print(f"Completion Tokens:     {measurement.completion_tokens}")
        print(f"Reasoning Tokens:      {reasoning}")
        print(f"Non-Reasoning Tokens:  {non_reasoning}")
        print(f"Latency:               {latency}")
        print(f"Estimated Cost:        {cost}")
        print(f"Finish Reason:         {measurement.finish_reason}")
    else:
        print(f"FAILED: {measurement.error}")


def _print_condition_summary(condition_id: str, measurements_for_condition: list[EffortMeasurement]) -> None:
    from labs.lab01_token_anatomy.effort_stats import compute_descriptive_stats

    successes = [m for m in measurements_for_condition if m.succeeded]
    reasoning_stats = compute_descriptive_stats([m.reasoning_tokens for m in successes])
    completion_stats = compute_descriptive_stats([m.completion_tokens for m in successes])
    latency_stats = compute_descriptive_stats([m.total_latency_ms for m in successes])
    cost_stats = compute_descriptive_stats([m.estimated_cost_usd for m in successes])

    def _fmt(value):
        return f"{value:.2f}" if value is not None else "N/A"

    print(f"\n-- {condition_id} summary ({len(successes)}/{len(measurements_for_condition)} succeeded) --")
    print(f"  Mean Reasoning Tokens:  {_fmt(reasoning_stats.mean)}")
    print(f"  Mean Completion Tokens: {_fmt(completion_stats.mean)}")
    print(f"  Mean Latency (ms):      {_fmt(latency_stats.mean)}")
    print(f"  Mean Estimated Cost:    {_fmt(cost_stats.mean)}")


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


def run_effort(args: argparse.Namespace) -> None:
    requested_live = bool(args.live) and not args.dry_run
    probe_ctx = build_context(requested_live=False, config_path=args.config)

    effort_config = probe_ctx.config.lab01_effort
    if effort_config is None:
        print(
            "Lab01 effort mode requires a [lab01.effort] section in your config file -- see "
            "config/default.toml for a documented example to add to your own config (e.g. "
            "config/live.toml)."
        )
        return
    if not effort_config.enabled:
        print("Lab01 effort mode is disabled for this config (lab01.effort.enabled = false).")
        return

    model = args.model or probe_ctx.config.groq.default_model
    prompts = args.prompt or list(effort_config.prompts)
    reasoning_efforts = args.reasoning_effort or list(effort_config.reasoning_efforts)
    repetitions = args.repetitions or effort_config.repetitions

    # Validated against the globally known prompt/effort sets, NOT against
    # effort_config.prompts/reasoning_efforts: the config's own lists are
    # only this run's DEFAULT (e.g. just short_prompt, for the 6-call
    # default experiment) -- --prompt/--reasoning-effort must still be able
    # to select any of this project's valid values (e.g. `--prompt
    # short_prompt long_prompt` for the 12-call extended experiment)
    # without editing the config file. max_api_calls/max_estimated_cost_usd
    # are what actually bound how far a CLI override can expand the scope.
    unknown_prompts = [p for p in prompts if p not in EXPERIMENTS]
    if unknown_prompts:
        print(f"--prompt value(s) {unknown_prompts} are not valid prompt types {sorted(EXPERIMENTS.keys())}; nothing executed.")
        return
    unknown_efforts = [e for e in reasoning_efforts if e not in SUPPORTED_REASONING_EFFORTS]
    if unknown_efforts:
        print(
            f"--reasoning-effort value(s) {unknown_efforts} are not valid reasoning_effort "
            f"levels {list(SUPPORTED_REASONING_EFFORTS)}; nothing executed."
        )
        return

    effective_effort_config = replace(
        effort_config, prompts=tuple(prompts), reasoning_efforts=tuple(reasoning_efforts), repetitions=repetitions
    )

    try:
        conditions = build_conditions(prompts, reasoning_efforts)
    except EffortMatrixError as exc:
        print(f"Invalid effort configuration: {exc}")
        return

    plan = build_execution_plan(
        conditions,
        repetitions,
        effort_config.randomize_execution_order,
        effort_config.randomization_seed,
        effort_config.use_paired_api_seed,
    )

    _print_plan_header(model, conditions, plan, live_enabled=requested_live)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = effort_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab01 effort mode requires --live (and execution.live_api_enabled=true) to "
            "make real measurements against the Groq API; nothing was executed. Use --dry-run "
            "to preview the plan without --live."
        )
        return

    ctx = build_context(requested_live=True, config_path=args.config)

    try:
        validate_resume_fingerprint(run_dir, effective_effort_config, model)
    except ResumeConfigurationMismatch as exc:
        print(f"\n{exc}")
        return

    existing_measurements = read_existing_measurements(run_dir)
    already_done = completed_pairs(existing_measurements)
    remaining_plan = [c for c in plan if (c.condition.condition_id, c.repetition_index) not in already_done]

    manifest = read_manifest(run_dir)
    if manifest is None:
        manifest = build_initial_manifest(
            experiment_run_id, effective_effort_config, model, len(conditions), len(plan)
        )
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
        conservative_per_call_cost_ceiling(pricing, effort_config.max_completion_tokens, longest_prompt)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=effort_config.max_api_calls,
        max_estimated_cost_usd=effort_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )
    for prior in existing_measurements:
        budget.record_call(prior.estimated_cost_usd if prior.succeeded else None)

    completed_count_by_condition: dict[str, int] = {}
    for m in existing_measurements:
        if m.succeeded:
            completed_count_by_condition[m.condition_id] = completed_count_by_condition.get(m.condition_id, 0) + 1
    measurements_by_condition: dict[str, list[EffortMeasurement]] = {}
    for m in existing_measurements:
        measurements_by_condition.setdefault(m.condition_id, []).append(m)

    stopped_reason = None
    measurements: list[EffortMeasurement] = list(existing_measurements)
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
            effort_config.max_completion_tokens,
            experiment_run_id,
            planned_call,
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        append_measurement(run_dir, measurement, is_first_write=(len(measurements) == 0))
        measurements.append(measurement)
        measurements_by_condition.setdefault(measurement.condition_id, []).append(measurement)

        _print_call_progress(planned_call, measurement, repetitions)

        completed_count_by_condition[measurement.condition_id] = (
            completed_count_by_condition.get(measurement.condition_id, 0) + 1
        )
        if completed_count_by_condition[measurement.condition_id] == repetitions:
            _print_condition_summary(measurement.condition_id, measurements_by_condition[measurement.condition_id])

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    write_manifest(run_dir, manifest)

    planned_calls_by_condition = {c.condition_id: repetitions for c in conditions}
    summaries = summarize_all_effort_conditions(conditions, planned_calls_by_condition, measurements)
    write_condition_summary_csv(run_dir, summaries)

    write_effort_summary_json(
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

    if any(m.succeeded for m in measurements):
        from labs.lab01_token_anatomy.effort_charts import generate_effort_charts

        chart_paths = generate_effort_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")

    _print_completion_summary(measurements, len(plan), run_dir, stopped_reason)
