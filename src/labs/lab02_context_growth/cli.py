from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab02_context_growth.experiments import CONDITIONS, run_context_experiment, run_multi_turn_growth

LAB_ID = "lab02"

_ALL_EXPERIMENTS = [*CONDITIONS.keys(), "multi_turn_growth", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab02 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model
    repetitions = args.repetitions or ctx.config.experiment.default_repetitions
    all_results = []

    conditions_to_run = list(CONDITIONS.keys()) if args.experiment in ("all", "context_conditions") else (
        [args.experiment] if args.experiment != "multi_turn_growth" else []
    )
    for condition in conditions_to_run:
        print(f"Running lab02/{condition} on {model} x{repetitions}...")
        results = run_context_experiment(ctx.client, ctx.config, condition, model, repetitions)
        all_results.extend(results)
        for r in results:
            print(f"  prompt_tokens={r.tokens_and_cost.prompt_tokens} client_elapsed_ms={r.performance.client_elapsed_ms:.1f}")

    if args.experiment in ("all", "multi_turn_growth"):
        print(f"Running lab02/multi_turn_growth on {model} x{args.turns} turns...")
        results = run_multi_turn_growth(ctx.client, ctx.config, model, args.turns)
        all_results.extend(results)
        for r in results:
            print(f"  turn={r.metadata.model_parameters['turn']} prompt_tokens={r.tokens_and_cost.prompt_tokens}")

    output_name = args.output_name or "lab02_context_growth"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab02 - Context Growth & Context Bloat")
    parser.add_argument("experiment", choices=_ALL_EXPERIMENTS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--turns", type=int, default=5, help="Number of turns for multi_turn_growth.")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
