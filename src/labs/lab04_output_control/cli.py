from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab04_output_control.experiments import (
    run_multiple_invocations,
    run_output_limit_sweep,
    run_single_invocation,
    summarize_invocation_comparison,
)

LAB_ID = "lab04"

_EXPERIMENTS = ["output_limit_sweep", "invocation_comparison", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab04 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model
    all_results = []

    if args.experiment in ("output_limit_sweep", "all"):
        max_tokens_values = [16, 64, 256]
        print(f"Running lab04/output_limit_sweep on {model} for max_tokens={max_tokens_values}...")
        results = run_output_limit_sweep(ctx.client, ctx.config, model, max_tokens_values)
        all_results.extend(results)
        for r in results:
            print(f"  max_tokens={r.metadata.model_parameters['max_tokens']} completion_tokens={r.tokens_and_cost.completion_tokens}")

    if args.experiment in ("invocation_comparison", "all"):
        print(f"Running lab04/invocation_comparison on {model}...")
        single_results = run_single_invocation(ctx.client, ctx.config, model)
        multi_results = run_multiple_invocations(ctx.client, ctx.config, model)
        all_results.extend(single_results)
        all_results.extend(multi_results)
        summary = summarize_invocation_comparison(single_results, multi_results)
        print(f"  single_invocation: {summary['single_invocation']}")
        print(f"  multiple_invocations: {summary['multiple_invocations']}")

    output_name = args.output_name or "lab04_output_control"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab04 - Output Control & Invocation Optimization")
    parser.add_argument("experiment", choices=_EXPERIMENTS)
    parser.add_argument("--model", default=None)
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
