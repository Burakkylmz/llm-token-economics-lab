"""Markdown analysis report for one Lab07 simulation run.

Mirrors every other redesigned lab's report module structure, with an
explicit, required limitations section restating this lab's "best
strategy is scenario-dependent" teaching principle rather than declaring
one universal winner.
"""

from __future__ import annotations

from typing import Sequence

from labs.lab07_rate_limiting.stats import StrategySummary, group_by_scenario


def _fmt(value, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "N/A"


def build_analysis_report_markdown(manifest: dict, summaries: Sequence[StrategySummary]) -> str:
    lines: list[str] = []
    lines.append(f"# Lab07 Rate Limiting & Retry Strategies Report -- {manifest.get('experiment_run_id', 'unknown-run')}")
    lines.append("")

    lines.append("## 1. Simulation Configuration")
    lines.append("")
    lines.append("**This lab makes zero real Groq API calls, in every mode.**")
    lines.append("")
    lines.append(f"- Profile: `{manifest.get('profile')}`")
    lines.append(f"- Randomization seed: {manifest.get('randomization_seed')}")
    lines.append(f"- Simulation version: `{manifest.get('simulation_version')}`")
    lines.append(f"- Planned simulation runs: {manifest.get('planned_simulation_runs')}")
    lines.append(f"- real_api_calls: {manifest.get('real_api_calls')}")
    lines.append(f"- real_token_usage: {manifest.get('real_token_usage')}")
    lines.append(f"- real_api_cost: {manifest.get('real_api_cost')}")
    lines.append("")

    grouped = group_by_scenario(summaries)
    for scenario_id, scenario_summaries in grouped.items():
        lines.append(f"## Scenario: `{scenario_id}`")
        lines.append("")
        lines.append("| strategy | success rate | total attempts | total 429s | mean wait (s) | max wait (s) | completion time (s) | collisions |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for s in scenario_summaries:
            lines.append(
                f"| {s.strategy_id} | {_fmt_pct(s.success_rate)} ({s.successful_clients}/{s.total_clients}) | "
                f"{s.total_attempts} | {s.total_429_responses} | {_fmt(s.mean_wait_seconds)} | {_fmt(s.max_wait_seconds)} | "
                f"{_fmt(s.completion_time_seconds)} | {s.retry_collision_count} |"
            )
        lines.append("")

    lines.append("## Reliability vs Efficiency")
    lines.append("")
    lines.append(
        "Success rate/recovery time (reliability) and total attempts/429 responses/wait time (efficiency) are "
        "reported SEPARATELY above -- the strategy with the lowest total wait time is never, by that fact alone, "
        "declared the best strategy. Which strategy is preferable depends on the scenario and on which of these "
        "two goals matters more for a given production constraint."
    )
    lines.append("")

    lines.append("## Experimental Limitations")
    lines.append("")
    lines.append("- All measurements here are SYNTHETIC simulation data -- not real Groq API telemetry, not real token usage, not real cost.")
    lines.append("- Jitter's seeded draws describe this one run's seed only; a different seed can produce a different trajectory (this is expected, not a bug).")
    lines.append("- `retry_collision_count` is only meaningful for multi-client scenarios -- it is genuinely 0 (not a placeholder) for every single-client scenario.")
    lines.append("- This simulation's server model is a deliberately simplified, documented approximation of real rate-limiting behavior, never a claim to replicate Groq's actual production implementation.")
    lines.append("- No strategy is assumed or asserted to win in every scenario -- different scenarios are expected to favor different strategies.")
    lines.append("")

    return "\n".join(lines)
