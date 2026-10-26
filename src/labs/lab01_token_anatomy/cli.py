from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.metrics import ExperimentResult
from lab_core.results_writer import append_csv, write_json
from labs.lab01_token_anatomy.experiments import EXPERIMENTS, run_experiment
from labs.lab01_token_anatomy.factorial_cli import add_factorial_arguments, run_factorial
from labs.lab01_token_anatomy.observability import summarize_values, tally_finish_reasons

LAB_ID = "lab01"

# Below this many repetitions, a variability summary is descriptive of this
# one run only -- flag it so nobody reads it as a statistically robust result.
SMALL_SAMPLE_REPETITIONS = 10


def _fmt_number(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _fmt_mean(value) -> str:
    return f"{value:.1f}" if value is not None else "N/A"


def _fmt_stdev(value, count_available: int) -> str:
    if value is not None:
        return f"{value:.2f}"
    return "N/A (need >=2 measurements)" if count_available < 2 else "N/A"


def _fmt_cost(value) -> str:
    return f"${value:.8f}" if value is not None else "N/A"


def _print_run(index: int, total: int, result: ExperimentResult) -> None:
    tokens = result.tokens_and_cost
    derived = result.derived
    print(f"\nRun {index}/{total}")
    print(f"  prompt_tokens:         {_fmt_number(tokens.prompt_tokens)}")
    print(f"  completion_tokens:     {_fmt_number(tokens.completion_tokens)}")
    print(f"  reasoning_tokens:      {_fmt_number(tokens.reasoning_tokens)}")
    print(f"  non_reasoning_tokens:  {_fmt_number(derived.non_reasoning_completion_tokens)}")
    finish_reason = tokens.finish_reason or "N/A"
    note = ""
    if finish_reason == "length":
        note = "  (possible truncation -- stopped at a generation limit, not a natural stop)"
    print(f"  finish_reason:         {finish_reason}{note}")
    print(f"  estimated_cost_usd:    {_fmt_cost(tokens.estimated_cost_usd)}")


def _print_and_build_summary(experiment: str, model: str, results: list[ExperimentResult]) -> dict:
    prompt_token_values = {r.tokens_and_cost.prompt_tokens for r in results}
    prompt_tokens_per_call = next(iter(prompt_token_values)) if len(prompt_token_values) == 1 else "varies"

    completion_stats = summarize_values([r.tokens_and_cost.completion_tokens for r in results])
    reasoning_stats = summarize_values([r.tokens_and_cost.reasoning_tokens for r in results])
    finish_reason_tally = tally_finish_reasons([r.tokens_and_cost.finish_reason for r in results])

    title = experiment.replace("_", " ").title()
    print(f"\nLAB01 -- {title} Summary\n")
    print(f"Calls:                       {len(results)}")
    print(f"Prompt Tokens per Call:      {prompt_tokens_per_call}")

    print("\nCompletion Tokens:")
    print(f"  Minimum:                  {_fmt_number(completion_stats.minimum)}")
    print(f"  Maximum:                  {_fmt_number(completion_stats.maximum)}")
    print(f"  Mean:                     {_fmt_mean(completion_stats.mean)}")
    print(f"  Standard Deviation:       {_fmt_stdev(completion_stats.standard_deviation, completion_stats.count_available)}")
    print(f"  Range:                    {_fmt_number(completion_stats.range)}")

    print("\nReasoning Tokens:")
    print(f"  Mean:                     {_fmt_mean(reasoning_stats.mean)}")
    print(f"  Available Measurements:   {reasoning_stats.count_available}/{reasoning_stats.count_total}")

    print("\nFinish Reasons:")
    for reason, count in sorted(finish_reason_tally.items()):
        print(f"  {reason}:{' ' * max(1, 22 - len(reason))}{count}")

    if len(results) < SMALL_SAMPLE_REPETITIONS:
        print(
            f"\nNote: only {len(results)} repetition(s) -- not a statistically robust "
            "sample; treat this variability summary as descriptive of this run, not "
            "as a general claim about the model."
        )

    return {
        "experiment_id": f"lab01_{experiment}",
        "model": model,
        "calls": len(results),
        "prompt_tokens_per_call": prompt_tokens_per_call,
        "completion_tokens": asdict(completion_stats),
        "reasoning_tokens": asdict(reasoning_stats),
        "finish_reasons": finish_reason_tally,
        "standard_deviation_method": "sample (statistics.stdev, n-1 denominator)",
        "statistically_robust": len(results) >= SMALL_SAMPLE_REPETITIONS,
    }


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "factorial":
        run_factorial(args)
        return

    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab01 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model
    repetitions = args.repetitions or ctx.config.experiment.default_repetitions
    experiment_names = list(EXPERIMENTS.keys()) if args.experiment == "all" else [args.experiment]

    all_results: list[ExperimentResult] = []
    summaries: list[dict] = []
    for experiment in experiment_names:
        print(f"Running lab01/{experiment} on {model} x{repetitions}...")
        results = run_experiment(ctx.client, ctx.config, experiment, model, repetitions)
        all_results.extend(results)
        for i, result in enumerate(results, start=1):
            _print_run(i, len(results), result)
        summaries.append(_print_and_build_summary(experiment, model, results))

    output_name = args.output_name or "lab01_token_anatomy"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")

    summary_path = ctx.output_dir / f"{output_name}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, ensure_ascii=False)

    print(f"\nResults written to {ctx.output_dir}/{output_name}.json, .csv, and _summary.json")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab01 - Token Anatomy & Cost Calculation")
    parser.add_argument(
        "experiment",
        choices=[*EXPERIMENTS.keys(), "all", "factorial"],
        help="short_prompt/long_prompt/all run Lab01's basic mode; factorial runs the "
        "full-factorial reasoning_effort x temperature x top_p experiment engine.",
    )
    parser.add_argument("--model", default=None, help="Groq model id (default: config groq.default_model).")
    add_common_arguments(parser)
    add_factorial_arguments(parser)
    parser.set_defaults(func=_run)
