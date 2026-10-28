"""CLI wiring and orchestration for Lab07 -- Rate Limiting & Retry
Strategies.

LAB07 IS SIMULATION ONLY: it never constructs a Groq client, never needs
GROQ_API_KEY, and never makes a real HTTP request in ANY mode -- `--live`
is accepted only for backward compatibility with the pre-redesign CLI and
is always ignored (with an explicit, visible warning), never silently.

Execution modes (the `experiment` positional argument) -- backward
compatible with the pre-redesign commands:
  - no_backoff / exponential_backoff / exponential_backoff_with_jitter /
    rate_limit_aware: runs only that one strategy, across `--profile`'s
    scenarios (or `--scenario`'s one override).
  - all: every strategy -- this project's documented 12-run default (3
    scenarios x 4 strategies).
  - backfill: offline chart (re)generation from existing results.

`--fail-count`/`--retry-after-seconds` remain supported exactly as
before: passing either forces a single-scenario legacy run (Scenario 1,
Fixed Transient Failures) with that fail count and/or a fixed injected
Retry-After header, ignoring `--profile`/`--scenario`.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from lab_core.config import load_config
from labs.lab07_rate_limiting.charts_cli import add_backfill_arguments, run_backfill
from labs.lab07_rate_limiting.matrix import PROFILES, scenarios_for_profile
from labs.lab07_rate_limiting.models import STRATEGIES, PlannedSimulationRun
from labs.lab07_rate_limiting.persistence import (
    RunIdAlreadyUsed,
    build_initial_manifest,
    lab07_run_dir,
    refuse_if_already_has_measurements,
    write_analysis_report,
    write_attempt_timeline_csv,
    write_experiment_plan_csv,
    write_experiment_summary_json,
    write_manifest,
    write_raw_measurements,
    write_scenario_comparison_csv,
    write_strategy_summary_csv,
)
from labs.lab07_rate_limiting.report import build_analysis_report_markdown
from labs.lab07_rate_limiting.runner import simulate_scenario
from labs.lab07_rate_limiting.scenarios import EXTENDED_SCENARIOS, SCENARIO_TRANSIENT_FAILURES, build_scenario
from labs.lab07_rate_limiting.stats import group_by_scenario, summarize_strategy

LAB_ID = "lab07"
_ALL_EXPERIMENTS = [*STRATEGIES, "all", "backfill"]


def add_lab07_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None, help="Path to a TOML config file (default: config/default.toml).")
    parser.add_argument("--live", action="store_true", help="Ignored: Lab07 never makes real API calls, in any mode.")
    parser.add_argument("--output-name", type=str, default=None, help="Unused (kept for CLI-flag backward compatibility).")
    parser.add_argument("--profile", default=None, choices=PROFILES, help="Scenario scope (default: config lab07.default_profile).")
    parser.add_argument("--scenario", default=None, choices=EXTENDED_SCENARIOS, help="Override --profile: run only this one scenario.")
    parser.add_argument("--seed", type=int, default=None, dest="seed", help="Override lab07.randomization_seed.")
    parser.add_argument("--max-simulation-events", type=int, default=None, dest="max_simulation_events")
    parser.add_argument("--fail-count", type=int, default=None, dest="fail_count", help="Legacy: forces a single-scenario (transient_failures) run with this fail count.")
    parser.add_argument("--retry-after-seconds", type=str, default=None, dest="retry_after_seconds", help="Legacy: injects this fixed Retry-After header into transient_failures' failing responses.")
    parser.add_argument(
        "--experiment-run-id", default=None,
        help="Name this run (default: an auto-generated timestamp-based id). Lab07 does "
        "not support resuming a run -- reusing an id that already has measurements is refused.",
    )


def _generate_run_id() -> str:
    return datetime.now(timezone.utc).strftime("lab07-%Y%m%dT%H%M%SZ")


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def _print_plan_header(profile, scenario_ids, strategies, total_runs, seed, max_simulation_events) -> None:
    print("LAB07 -- Rate Limiting & Retry Strategies\n")
    print("Execution Mode: SIMULATION")
    print("Real API Calls: 0\n")
    print(f"Profile: {profile}")
    print(f"Scenarios: {len(scenario_ids)} ({', '.join(scenario_ids)})")
    print(f"Strategies: {len(strategies)} ({', '.join(strategies)})\n")
    print(f"Planned Simulation Runs: {total_runs}")
    print("Virtual Clock: Enabled")
    print(f"Random Seed: {seed}")
    print(f"Max Simulation Events: {max_simulation_events}")


def _print_strategy_result(scenario_id, strategy_id, summary, records) -> None:
    print(f"\nScenario: {scenario_id}")
    print(f"Strategy: {strategy_id}\n")
    print(f"Attempts:                 {summary.total_attempts}")
    print(f"Retries:                  {summary.total_retry_count}")
    print(f"HTTP 429 Responses:       {summary.total_429_responses}")
    print(f"Succeeded:                {summary.successful_clients}/{summary.total_clients}")
    print(f"Success Rate:             {_fmt_pct(summary.success_rate)}")
    print(f"\nTotal Scheduled Wait:     {_fmt(summary.total_scheduled_wait_seconds)} seconds")
    print(f"Virtual Completion Time:  {_fmt(summary.completion_time_seconds)} seconds")
    print(f"Retry Collisions:         {summary.retry_collision_count}")

    client0_records = sorted((r for r in records if r.client_id == 0), key=lambda r: r.attempt_number)
    if client0_records:
        print("\nRetry Timeline (client 0):")
        for r in client0_records:
            print(f"  T={r.virtual_timestamp_seconds:.2f}s  Attempt {r.attempt_number} -> {r.http_status}" + (f"  (outcome={r.final_outcome})" if r.final_outcome else ""))


def _print_scenario_summary(scenario_id, summaries_for_scenario) -> None:
    print(f"\n-- {scenario_id} Summary --\n")
    for s in summaries_for_scenario:
        print(f"{s.strategy_id}:")
        print(f"  Success Rate:           {_fmt_pct(s.success_rate)}")
        print(f"  Total Attempts:         {s.total_attempts}")
        print(f"  HTTP 429 Count:         {s.total_429_responses}")
        print(f"  Recovery Time:          {_fmt(s.completion_time_seconds)} s\n")


def _print_final_assessment(summaries) -> None:
    print("\n-- Final Assessment (SIMULATION ONLY) --\n")
    grouped = group_by_scenario(summaries)
    for scenario_id, scenario_summaries in grouped.items():
        best_reliability = max(scenario_summaries, key=lambda s: (s.success_rate or 0))
        best_efficiency = min(scenario_summaries, key=lambda s: s.total_scheduled_wait_seconds)
        print(f"{scenario_id}:")
        print(f"  Reliability Comparison:          highest success rate = {best_reliability.strategy_id} ({_fmt_pct(best_reliability.success_rate)})")
        print(f"  Retry Traffic / Wait Comparison:  lowest total wait = {best_efficiency.strategy_id} ({_fmt(best_efficiency.total_scheduled_wait_seconds)}s)")
        if best_reliability.strategy_id != best_efficiency.strategy_id:
            print("  Note: the most reliable strategy and the lowest-wait strategy are NOT the same -- the lowest wait time is never, by itself, evidence of the best overall strategy.")
    print("\nAll figures above are synthetic simulation data (real_api_calls=0).")


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "backfill":
        run_backfill(args)
        return

    config = load_config(args.config)
    lab07_config = config.lab07

    print("LAB07 IS SIMULATION ONLY.")
    print("No real API calls will be made.\n")
    if args.live:
        print("[NOTE] --live was passed but is IGNORED: Lab07 never makes real API calls, in any mode.\n")

    if not lab07_config.enabled:
        print("Lab07 is disabled for this config (lab07.enabled = false).")
        return

    profile = args.profile or lab07_config.default_profile
    seed = args.seed if args.seed is not None else lab07_config.randomization_seed
    max_simulation_events = args.max_simulation_events or lab07_config.max_simulation_events

    legacy_single_scenario = args.fail_count is not None or args.retry_after_seconds is not None
    if legacy_single_scenario:
        scenario_ids = [SCENARIO_TRANSIENT_FAILURES]
    elif args.scenario is not None:
        scenario_ids = [args.scenario]
    else:
        scenario_ids = list(scenarios_for_profile(profile))

    strategies = [args.experiment] if args.experiment in STRATEGIES else list(STRATEGIES)

    specs = {}
    for scenario_id in scenario_ids:
        specs[scenario_id] = build_scenario(
            scenario_id,
            transient_fail_count=args.fail_count if args.fail_count is not None else lab07_config.transient_fail_count,
            recovery_time_seconds=lab07_config.recovery_time_seconds,
            contention_client_count=lab07_config.contention_client_count,
            contention_capacity_per_window=lab07_config.contention_capacity_per_window,
            contention_window_seconds=lab07_config.contention_window_seconds,
            quota_client_count=lab07_config.quota_client_count,
            quota_rpm=lab07_config.quota_rpm,
            quota_tpm=lab07_config.quota_tpm,
            quota_token_demand_per_request=lab07_config.quota_token_demand_per_request,
            quota_window_seconds=lab07_config.quota_window_seconds,
        )
        if legacy_single_scenario and args.retry_after_seconds is not None:
            specs[scenario_id].server.fail_count_retry_after = args.retry_after_seconds

    for scenario_id, spec in specs.items():
        if len(spec.client_ids) > lab07_config.max_clients:
            print(f"\nRefusing: scenario '{scenario_id}' plans {len(spec.client_ids)} clients, exceeding lab07.max_clients ({lab07_config.max_clients}). Nothing was executed.")
            return

    total_planned_runs = len(scenario_ids) * len(strategies)
    experiment_run_id = args.experiment_run_id or _generate_run_id()
    run_dir = lab07_run_dir(Path(config.experiment.results_dir), experiment_run_id)

    _print_plan_header(profile, scenario_ids, strategies, total_planned_runs, seed, max_simulation_events)

    try:
        refuse_if_already_has_measurements(run_dir)
    except RunIdAlreadyUsed as exc:
        print(f"\n{exc}")
        return

    manifest = build_initial_manifest(experiment_run_id, lab07_config, profile, total_planned_runs, seed)
    write_manifest(run_dir, manifest)

    plan: list[PlannedSimulationRun] = []
    all_records = []
    all_client_results = []
    summaries = []
    index = 1

    for scenario_id in scenario_ids:
        spec = specs[scenario_id]
        scenario_summaries = []
        for strategy_id in strategies:
            plan.append(PlannedSimulationRun(index, scenario_id, strategy_id, spec.client_ids))
            simulation_run_id = f"{experiment_run_id}-{scenario_id}-{strategy_id}"
            rng = random.Random(seed)
            records, client_results = simulate_scenario(
                spec, strategy_id, simulation_run_id, lab07_config.max_attempts, lab07_config.base_delay_seconds,
                lab07_config.max_delay_seconds, lab07_config.max_total_wait_seconds, lab07_config.simulation_deadline_seconds,
                rng, max_simulation_events,
            )
            all_records.extend(records)
            all_client_results.extend(client_results)
            summary = summarize_strategy(scenario_id, strategy_id, client_results, records)
            summaries.append(summary)
            scenario_summaries.append(summary)
            _print_strategy_result(scenario_id, strategy_id, summary, records)
            index += 1
        _print_scenario_summary(scenario_id, scenario_summaries)

    write_experiment_plan_csv(run_dir, plan)
    write_attempt_timeline_csv(run_dir, all_records)
    write_raw_measurements(run_dir, all_client_results)
    write_strategy_summary_csv(run_dir, summaries)
    write_scenario_comparison_csv(run_dir, summaries)

    manifest["execution_completed_at"] = datetime.now(timezone.utc).isoformat()
    write_manifest(run_dir, manifest)

    write_experiment_summary_json(
        run_dir,
        {
            "experiment_run_id": experiment_run_id,
            "profile": profile,
            "randomization_seed": seed,
            "real_api_calls": 0,
            "real_token_usage": None,
            "real_api_cost": 0,
            "strategy_summaries": [asdict(s) for s in summaries],
            "attempt_records": [asdict(r) for r in all_records],
        },
    )

    report_markdown = build_analysis_report_markdown(manifest, summaries)
    write_analysis_report(run_dir, report_markdown)

    _print_final_assessment(summaries)

    try:
        from labs.lab07_rate_limiting.charts import generate_retry_charts

        chart_paths = generate_retry_charts(run_dir)
        if chart_paths:
            print(f"\nCharts written: {len(chart_paths)} -> {run_dir / 'charts'}")
    except Exception as exc:  # noqa: BLE001 -- a charting failure must never cost an already-persisted measurement.
        print(f"\nChart generation failed (results above are unaffected): {exc}")

    print(f"\nResults Directory: {run_dir}")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab07 - Rate Limiting & Retry Strategies (fully simulated)")
    parser.add_argument(
        "experiment",
        choices=_ALL_EXPERIMENTS,
        help="A single strategy name runs only that strategy; all runs every strategy (this project's "
        "documented 12-run default: 3 scenarios x 4 strategies); backfill (re)generates PNG charts "
        "from existing results, offline. LAB07 never makes a real API call in any mode.",
    )
    add_lab07_arguments(parser)
    add_backfill_arguments(parser)
    parser.set_defaults(func=_run)
