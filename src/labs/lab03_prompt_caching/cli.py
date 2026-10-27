"""CLI wiring and orchestration for Lab03 -- Prompt Caching.

Execution modes (the `experiment` positional argument):
  - repeated_prefix / cache_unfriendly: run just that one condition,
    `--repetitions` times (cycling through the same paired questions as
    `all` would, for cross-run comparability).
  - all: both conditions, PAIRED and INTERLEAVED (pair 1: repeated_prefix,
    cache_unfriendly; pair 2: ...; never all-of-one-condition-then-all-of-
    the-other) -- this project's documented 6-call default (3 repetitions
    x 2 conditions).
  - backfill: offline, read-only chart (re)generation from existing
    results -- see labs.lab03_prompt_caching.charts_cli.

Every live-executing mode writes into its own
`results/lab03_prompt_caching/<experiment_run_id>/` directory and never
touches the pre-existing flat `results/lab03_prompt_caching.json`/`.csv`
files from before this redesign.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone

from lab_core.lab_cli_support import add_common_arguments, build_context
from labs.lab03_prompt_caching.cache_probe import build_probe_plan, verify_probe_plan_is_identical
from labs.lab03_prompt_caching.cache_status import CACHE_HIT, CACHE_MISS, CACHE_USAGE_UNAVAILABLE, INVALID_USAGE
from labs.lab03_prompt_caching.charts_cli import add_backfill_arguments, run_backfill
from labs.lab03_prompt_caching.experiments import (
    CONDITION_ORDER,
    CONDITION_REPEATED_PREFIX,
    STATIC_KNOWLEDGE_BASE,
    STATIC_RESPONSE_RULES,
)
from labs.lab03_prompt_caching.matrix import build_cache_plan
from labs.lab03_prompt_caching.models import CacheMeasurement
from labs.lab03_prompt_caching.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab03_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_cache_diagnostics,
    write_cache_measurements,
    write_condition_summary_csv,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
)
from labs.lab03_prompt_caching.report import build_analysis_report_markdown
from labs.lab03_prompt_caching.runner import Budget, conservative_per_call_cost_ceiling, execute_planned_cache_call
from labs.lab03_prompt_caching.stats import summarize_all_cache_conditions

LAB_ID = "lab03"

CACHE_PROBE_EXPERIMENT = "cache_probe"
_ALL_EXPERIMENTS = [*CONDITION_ORDER, "all", CACHE_PROBE_EXPERIMENT, "backfill"]

# Both conditions share the same static knowledge base + rules length, so
# either is an equally conservative stand-in for the pre-flight cost
# ceiling (see _run below).
_LONGEST_STATIC_CONTENT = STATIC_KNOWLEDGE_BASE + STATIC_RESPONSE_RULES


def add_lab03_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", default=None, help="Must support prompt caching (see config lab03.caching_capable_models).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print and save the experiment plan, never call the API (always safe: "
        "ignores --live, needs no API key).",
    )
    parser.add_argument(
        "--experiment-run-id",
        default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab03 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab03-%Y%m%dT%H%M%SZ")


def _conditions_for(experiment: str) -> list[str]:
    if experiment == "all":
        return list(CONDITION_ORDER)
    if experiment in CONDITION_ORDER:
        return [experiment]
    return []


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


def _print_plan_header(model: str, conditions: list[str], repetitions: int, total_planned: int, live_enabled: bool) -> None:
    print("LAB03 -- Prompt Caching\n")
    print(f"Model: {model}\n")
    print("Conditions:")
    for condition_id in conditions:
        print(f"  {condition_id}")
    print()
    print(f"Repetitions: {repetitions}\n")
    print(f"Planned API Calls: {total_planned}\n")
    print(f"Live API Execution: {'Enabled' if live_enabled else 'Disabled'}")


def _print_call_progress(measurement: CacheMeasurement, repetitions: int) -> None:
    print(f"\nCondition: {measurement.condition_id}")
    print(f"Run: {measurement.pair_index}/{repetitions}\n")
    if not measurement.succeeded:
        print(f"FAILED: {measurement.error}")
        return
    print(f"Prompt Tokens:          {_fmt(measurement.prompt_tokens, 0)}")
    print(f"Cached Tokens:          {_fmt(measurement.cached_tokens, 0)}")
    print(f"Uncached Prompt Tokens: {_fmt(measurement.uncached_prompt_tokens, 0)}")
    print(f"Cache Hit:              {measurement.request_cache_hit}")
    print(f"Cache Status:           {measurement.cache_status}")
    print(f"Cache Ratio:            {_fmt_pct(measurement.request_cache_ratio)}")
    print(f"Completion Tokens:      {_fmt(measurement.completion_tokens, 0)}")
    print(f"Reasoning Tokens:       {_fmt(measurement.reasoning_tokens, 0)}")
    print(f"Latency:                {_fmt(measurement.total_latency_ms)} ms")
    print(f"Estimated Cost:         {_fmt_cost(measurement.estimated_cost_usd)}")
    print(f"Estimated Cache Savings:{_fmt_cost(measurement.estimated_cache_savings_usd)}")
    print(f"Finish Reason:          {measurement.finish_reason}")
    print(f"Answer Correct:         {measurement.answer_correct} ({measurement.answer_evaluation_status})")


def _print_probe_stability(plan, model: str, max_completion_tokens: int) -> None:
    stability = verify_probe_plan_is_identical(plan, model, max_completion_tokens)
    print("\nPrefix Stability Verification (offline, computed before any API call):")
    print(f"  Stable Prefix Length: {stability['stable_prefix_length']} chars")
    print(f"  Full Prompt Length:   {stability['full_prompt_length']} chars")
    print(f"  Request Hash:         {stability['request_hash']}")
    print(
        f"  All {stability['probe_calls']} planned requests byte-identical: {stability['stable']} "
        "(hash equality is a construction check, NOT evidence of a real cache hit)"
    )


def _summarize_cache_verification(measurements: list[CacheMeasurement]) -> tuple[str, str]:
    """Returns (verification_line, reason_line) for the Level 2 live cache
    verification block. Never claims OBSERVED without at least one
    successful call whose cache_status is literally CACHE_HIT."""
    successes = [m for m in measurements if m.succeeded]
    hits = [m for m in successes if m.cache_status == CACHE_HIT]
    if hits:
        total_cached = sum(m.cached_tokens for m in hits if m.cached_tokens is not None)
        return "OBSERVED", f"{len(hits)}/{len(successes)} successful call(s) reported cached_tokens > 0 (total {total_cached} cached tokens)."

    reported = [m for m in successes if m.prompt_tokens_details_present]
    if reported:
        return (
            "NOT OBSERVED",
            f"{len(reported)}/{len(successes)} successful call(s) DID report prompt_tokens_details, but with "
            "cached_tokens == 0 (CACHE_MISS) on every one -- caching was evaluated by the provider but did not hit.",
        )

    invalid = [m for m in successes if m.cache_status == INVALID_USAGE]
    if invalid:
        return "NOT OBSERVED", f"{len(invalid)}/{len(successes)} successful call(s) reported inconsistent cache usage (cached_tokens > prompt_tokens) -- treat as unreliable, not a hit."

    return (
        "NOT OBSERVED",
        f"None of {len(successes)} successful call(s) had prompt_tokens_details in the provider response at all "
        "(CACHE_USAGE_UNAVAILABLE) -- this is a provider-side signal, not an SDK parsing bug (the installed SDK "
        "declares and preserves this field; see docs/instructor/README.md's Lab03 root-cause analysis).",
    )


def _print_condition_summary(condition_id: str, measurements: list[CacheMeasurement]) -> None:
    from labs.lab03_prompt_caching.cache_metrics import compute_observed_request_cache_hit_rate, compute_token_cache_ratio
    from labs.lab01_token_anatomy.factorial_stats import compute_descriptive_stats

    successes = [m for m in measurements if m.succeeded]
    correct = sum(1 for m in successes if m.answer_correct is True)
    latency_stats = compute_descriptive_stats([m.total_latency_ms for m in successes])
    cached_stats = compute_descriptive_stats([m.cached_tokens for m in successes])
    costs = [m.estimated_cost_usd for m in successes if m.estimated_cost_usd is not None]
    savings = [m.estimated_cache_savings_usd for m in successes if m.estimated_cache_savings_usd is not None]

    print(f"\n-- {condition_id} summary --\n")
    print(f"Successful Calls:          {len(successes)}/{len(measurements)}")
    print(f"Observed Request Hit Rate: {_fmt_pct(compute_observed_request_cache_hit_rate(successes))}")
    print(f"Token Cache Ratio:         {_fmt_pct(compute_token_cache_ratio(successes))}")
    print(f"Total Cached Tokens:       {_fmt(cached_stats.mean * cached_stats.count if cached_stats.mean is not None else None, 0)}")
    print(f"Mean Client Latency:       {_fmt(latency_stats.mean)} ms")
    print(f"Total Estimated Cost:      {_fmt_cost(sum(costs) if costs else None)}")
    print(f"Estimated Savings:         {_fmt_cost(sum(savings) if savings else None)}")
    print(f"Correct Answers:           {correct}/{len(successes)}")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    probe_ctx = build_context(requested_live=False, config_path=args.config)
    lab03_config = probe_ctx.config.lab03

    model = args.model or lab03_config.caching_capable_models[0]
    if model not in lab03_config.caching_capable_models:
        print(
            f"Warning: '{model}' is not in config lab03.caching_capable_models "
            f"{lab03_config.caching_capable_models}; prompt caching may not be supported, and "
            "the observed-hit-rate experiment would be misleading."
        )

    is_probe = args.experiment == CACHE_PROBE_EXPERIMENT
    if is_probe:
        repetitions = args.repetitions or lab03_config.cache_probe_calls
        plan = build_probe_plan(repetitions)
        conditions = [CACHE_PROBE_EXPERIMENT]
        # Every probe call's actual condition_id is CONDITION_REPEATED_PREFIX
        # (see labs.lab03_prompt_caching.cache_probe) -- stats/summary
        # grouping must key off that real id, not the display-only mode
        # name "cache_probe" used in the header above.
        condition_ids_for_summary = [CONDITION_REPEATED_PREFIX]
    else:
        repetitions = args.repetitions or lab03_config.repetitions
        conditions = _conditions_for(args.experiment)
        plan = build_cache_plan(
            conditions, repetitions, lab03_config.randomize_condition_order_within_pair, lab03_config.randomization_seed
        )
        condition_ids_for_summary = conditions
    total_planned = len(plan)

    requested_live = bool(args.live) and not args.dry_run
    _print_plan_header(model, conditions, repetitions, total_planned, live_enabled=requested_live)
    if is_probe:
        _print_probe_stability(plan, model, lab03_config.max_completion_tokens)

    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab03_run_dir(probe_ctx.output_dir, experiment_run_id)

    if args.dry_run:
        write_experiment_plan_csv(run_dir, plan)
        print(f"\nDry run only -- plan written to {run_dir / 'experiment_plan.csv'}. No API calls were made.")
        return

    if not requested_live:
        print(
            "\nLab03 requires --live (and execution.live_api_enabled=true) to make real "
            "measurements against the Groq API; nothing was executed. Use --dry-run to "
            "preview the plan without --live."
        )
        return

    if lab03_config.max_api_calls is not None and total_planned > lab03_config.max_api_calls:
        print(
            f"\nRefusing to execute: planned calls ({total_planned}) exceed "
            f"lab03.max_api_calls ({lab03_config.max_api_calls}). Lower --repetitions, or raise "
            "lab03.max_api_calls deliberately in your config if you intend more. No API calls were made."
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
        conservative_per_call_cost_ceiling(pricing, lab03_config.max_completion_tokens, _LONGEST_STATIC_CONTENT)
        if pricing is not None
        else 0.0
    )
    budget = Budget(
        max_api_calls=None,  # the pre-flight total_planned gate above already enforces this
        max_estimated_cost_usd=lab03_config.max_estimated_cost_usd,
        per_call_cost_ceiling=per_call_ceiling,
    )

    manifest = build_initial_manifest(experiment_run_id, lab03_config, model, total_planned)
    write_manifest(run_dir, manifest)
    write_experiment_plan_csv(run_dir, plan)

    measurements: list[CacheMeasurement] = []
    measurements_by_condition: dict[str, list[CacheMeasurement]] = {}
    completed_by_condition: dict[str, int] = {}
    stopped_reason = None

    for planned_call in plan:
        reason = budget.can_make_another_call()
        if reason:
            stopped_reason = reason
            print(f"\nStopping before call: {reason}")
            break

        measurement = execute_planned_cache_call(
            ctx.client, ctx.config.retry, model, pricing, lab03_config.max_completion_tokens, experiment_run_id, planned_call
        )
        budget.record_call(measurement.estimated_cost_usd if measurement.succeeded else None)
        measurements.append(measurement)
        measurements_by_condition.setdefault(measurement.condition_id, []).append(measurement)
        _print_call_progress(measurement, repetitions)

        completed_by_condition[measurement.condition_id] = completed_by_condition.get(measurement.condition_id, 0) + 1
        if completed_by_condition[measurement.condition_id] == repetitions:
            _print_condition_summary(measurement.condition_id, measurements_by_condition[measurement.condition_id])

    write_cache_measurements(run_dir, measurements)
    write_cache_diagnostics(run_dir, measurements)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat() if stopped_reason is None else None
    manifest["stopped_reason"] = stopped_reason
    manifest["experiment_mode"] = CACHE_PROBE_EXPERIMENT if is_probe else args.experiment
    write_manifest(run_dir, manifest)

    planned_calls_by_condition = {c: repetitions for c in condition_ids_for_summary}
    summaries = summarize_all_cache_conditions(condition_ids_for_summary, planned_calls_by_condition, measurements)
    write_condition_summary_csv(run_dir, summaries)

    successes = [m for m in measurements if m.succeeded]
    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "experiment_mode": CACHE_PROBE_EXPERIMENT if is_probe else args.experiment,
            "planned_calls": total_planned,
            "successful_calls": len(successes),
            "failed_calls": len(measurements) - len(successes),
            "stopped_reason": stopped_reason,
            "total_estimated_cost_usd": budget.accumulated_estimated_cost_usd,
            "conditions": [asdict(s) for s in summaries],
            # Per-call records, embedded here (rather than a third on-disk
            # file) so chart generation -- which only ever reads
            # experiment_summary.json -- has what it needs for the
            # request-indexed charts (cache ratio by request, cached
            # tokens vs latency) without re-reading cache_measurements.json.
            "measurements": [asdict(m) for m in measurements],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, measurements, summaries)
    write_analysis_report(run_dir, report_markdown)

    if successes:
        verification_status, verification_reason = _summarize_cache_verification(measurements)
        print(f"\nCache Verification: {verification_status}")
        print(f"Reason: {verification_reason}")

        # Best-effort, strictly after every measurement/report file above is
        # already written -- a charting failure must never cost an
        # already-persisted measurement.
        try:
            from labs.lab03_prompt_caching.charts import generate_cache_charts

            chart_paths = generate_cache_charts(run_dir)
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
    parser = subparsers.add_parser(LAB_ID, help="Lab03 - Prompt Caching")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="repeated_prefix/cache_unfriendly run one condition; all runs both, paired and "
        "interleaved (this project's documented 6-call default); cache_probe sends N (default "
        "3, see lab03.cache_probe_calls) sequential, byte-identical requests -- the strongest "
        "exact-prefix-match condition for observing a real cached_tokens > 0; backfill "
        "(re)generates PNG charts from existing results, offline, without any API key.",
    )
    add_common_arguments(parser)
    add_lab03_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
