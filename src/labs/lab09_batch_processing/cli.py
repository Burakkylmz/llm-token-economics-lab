from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab09_batch_processing.batch import download_batch_results, poll_batch_status, submit_batch_job
from labs.lab09_batch_processing.experiments import (
    PROMPTS,
    batch_output_to_results,
    prepare_batch_input_file,
    run_concurrent,
    run_sequential,
)

LAB_ID = "lab09"

_ACTIONS = [
    "sequential",
    "async",
    "prepare_batch_input",
    "submit_batch",
    "check_batch_status",
    "fetch_batch_results",
]


def _run(args: argparse.Namespace) -> None:
    if args.experiment == "prepare_batch_input":
        # Building the JSONL request file is pure local I/O; no API call, no --live needed.
        config_path = args.config
        from lab_core.config import load_config

        config = load_config(config_path)
        model = args.model or config.groq.default_model
        output_path = Path(config.experiment.results_dir) / "lab09_batch_input.jsonl"
        prepare_batch_input_file(model, output_path)
        print(f"Wrote {len(PROMPTS)} batch requests to {output_path}")
        return

    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "This lab09 action requires --live (and execution.live_api_enabled=true) "
            "to call the real Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model

    if args.experiment == "sequential":
        print(f"Running lab09/sequential on {model} ({len(PROMPTS)} prompts)...")
        results, summary = asyncio.run(run_sequential(ctx.async_client, ctx.config, model))
        print(f"  summary: {summary}")
        _write(ctx, results, args.output_name or "lab09_sequential")
        return

    if args.experiment == "async":
        print(f"Running lab09/async on {model} ({len(PROMPTS)} prompts, unbounded concurrency)...")
        results, summary = asyncio.run(run_concurrent(ctx.async_client, ctx.config, model, max_concurrent=len(PROMPTS)))
        print(f"  summary: {summary}")
        _write(ctx, results, args.output_name or "lab09_async")
        return

    if args.experiment == "submit_batch":
        input_path = Path(ctx.config.experiment.results_dir) / "lab09_batch_input.jsonl"
        if not input_path.exists():
            print(f"No batch input file at {input_path}. Run `lab09 prepare_batch_input` first.")
            return
        batch_id = submit_batch_job(ctx.client, input_path, ctx.config.lab09.batch_completion_window)
        print(f"Submitted batch job: {batch_id}")
        print("This does NOT wait for completion (can take minutes up to the completion window, e.g. 24h-7d).")
        print(f"Check status later with: lab-cli lab09 check_batch_status --batch-id {batch_id} --live")
        return

    if args.experiment == "check_batch_status":
        if not args.batch_id:
            print("--batch-id is required for check_batch_status.")
            return
        status = poll_batch_status(ctx.client, args.batch_id)
        print(f"Batch {args.batch_id}: {status}")
        return

    if args.experiment == "fetch_batch_results":
        if not args.output_file_id:
            print("--output-file-id is required for fetch_batch_results (from check_batch_status once completed).")
            return
        parsed_lines = download_batch_results(ctx.client, args.output_file_id)
        pricing = ctx.config.pricing.get(model)
        results = batch_output_to_results(parsed_lines, model, pricing)
        _write(ctx, results, args.output_name or "lab09_batch_api")
        return


def _write(ctx, results, output_name: str) -> None:
    write_json(results, ctx.output_dir / f"{output_name}.json")
    append_csv(results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(
        LAB_ID,
        help="Lab09 - Batch Processing (sync/async baselines vs the Groq Batch API)",
    )
    parser.add_argument("experiment", choices=_ACTIONS)
    parser.add_argument("--model", default=None)
    parser.add_argument("--batch-id", default=None, dest="batch_id")
    parser.add_argument("--output-file-id", default=None, dest="output_file_id")
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
