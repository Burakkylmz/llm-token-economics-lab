from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab05_model_routing.experiments import (
    run_fixed_model_baseline,
    run_rule_based_routing,
    summarize_quality_cost_latency,
)

LAB_ID = "lab05"

_EXPERIMENTS = ["fixed_model_baseline", "rule_based_routing", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab05 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    all_results = []

    if args.experiment in ("fixed_model_baseline", "all"):
        model = args.model or ctx.config.groq.default_model
        print(f"Running lab05/fixed_model_baseline on {model}...")
        results = run_fixed_model_baseline(ctx.client, ctx.config, model)
        all_results.extend(results)
        print(f"  summary: {summarize_quality_cost_latency(results)}")

    if args.experiment in ("rule_based_routing", "all"):
        print("Running lab05/rule_based_routing (small vs large model chosen per prompt)...")
        results = run_rule_based_routing(ctx.client, ctx.config)
        all_results.extend(results)
        print(f"  summary: {summarize_quality_cost_latency(results)}")

    output_name = args.output_name or "lab05_model_routing"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab05 - Model Routing")
    parser.add_argument("experiment", choices=_EXPERIMENTS)
    parser.add_argument("--model", default=None, help="Used only for fixed_model_baseline.")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
