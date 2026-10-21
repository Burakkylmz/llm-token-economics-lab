from __future__ import annotations

import argparse
import asyncio

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab08_concurrency.experiments import PROMPTS, run_concurrent, run_sequential

LAB_ID = "lab08"

_EXPERIMENTS = ["sequential", "concurrent", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab08 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model
    all_results = []

    if args.experiment in ("sequential", "all"):
        print(f"Running lab08/sequential on {model} ({len(PROMPTS)} prompts, one at a time)...")
        results, summary = asyncio.run(run_sequential(ctx.async_client, ctx.config, model))
        all_results.extend(results)
        print(f"  summary: {summary}")

    if args.experiment in ("concurrent", "all"):
        max_concurrent = args.max_concurrent or ctx.config.concurrency.max_concurrent_requests
        print(f"Running lab08/concurrent on {model} ({len(PROMPTS)} prompts, max_concurrent={max_concurrent})...")
        results, summary = asyncio.run(run_concurrent(ctx.async_client, ctx.config, model, max_concurrent))
        all_results.extend(results)
        print(f"  summary: {summary}")

    output_name = args.output_name or "lab08_concurrency"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab08 - Async Processing & Concurrency")
    parser.add_argument("experiment", choices=_EXPERIMENTS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-concurrent", type=int, default=None, dest="max_concurrent")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
