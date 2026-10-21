from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab01_token_anatomy.experiments import EXPERIMENTS, run_experiment

LAB_ID = "lab01"


def _run(args: argparse.Namespace) -> None:
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

    all_results = []
    for experiment in experiment_names:
        print(f"Running lab01/{experiment} on {model} x{repetitions}...")
        results = run_experiment(ctx.client, ctx.config, experiment, model, repetitions)
        all_results.extend(results)
        for r in results:
            print(
                f"  prompt_tokens={r.tokens_and_cost.prompt_tokens} "
                f"completion_tokens={r.tokens_and_cost.completion_tokens} "
                f"estimated_cost_usd={r.tokens_and_cost.estimated_cost_usd}"
            )

    output_name = args.output_name or "lab01_token_anatomy"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab01 - Token Anatomy & Cost Calculation")
    parser.add_argument("experiment", choices=[*EXPERIMENTS.keys(), "all"])
    parser.add_argument("--model", default=None, help="Groq model id (default: config groq.default_model).")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
