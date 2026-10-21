from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab03_prompt_caching.experiments import observed_cache_hit_rate, run_cache_unfriendly, run_repeated_prefix

LAB_ID = "lab03"

_EXPERIMENTS = ["repeated_prefix", "cache_unfriendly", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab03 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.lab03.caching_capable_models[0]
    if model not in ctx.config.lab03.caching_capable_models:
        print(
            f"Warning: '{model}' is not in config lab03.caching_capable_models "
            f"{ctx.config.lab03.caching_capable_models}; prompt caching may not be supported."
        )
    repetitions = args.repetitions or ctx.config.experiment.default_repetitions

    all_results = []
    if args.experiment in ("repeated_prefix", "all"):
        print(f"Running lab03/repeated_prefix on {model} x{repetitions}...")
        results = run_repeated_prefix(ctx.client, ctx.config, model, repetitions)
        all_results.extend(results)
        hit_rate = observed_cache_hit_rate(results)
        print(f"  observed cache hit rate: {hit_rate if hit_rate is not None else 'unavailable (no cached_tokens reported)'}")

    if args.experiment in ("cache_unfriendly", "all"):
        print(f"Running lab03/cache_unfriendly on {model} x{repetitions}...")
        results = run_cache_unfriendly(ctx.client, ctx.config, model, repetitions)
        all_results.extend(results)
        hit_rate = observed_cache_hit_rate(results)
        print(f"  observed cache hit rate: {hit_rate if hit_rate is not None else 'unavailable (no cached_tokens reported)'}")

    output_name = args.output_name or "lab03_prompt_caching"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab03 - Prompt Caching")
    parser.add_argument("experiment", choices=_EXPERIMENTS)
    parser.add_argument("--model", default=None, help="Must support prompt caching (see config lab03.caching_capable_models).")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
