from __future__ import annotations

import argparse

from lab_core.lab_cli_support import add_common_arguments, build_context
from lab_core.results_writer import append_csv, write_json
from labs.lab06_streaming.experiments import run_non_streaming, run_streaming

LAB_ID = "lab06"

_EXPERIMENTS = ["non_streaming", "streaming", "all"]


def _run(args: argparse.Namespace) -> None:
    ctx = build_context(requested_live=args.live, config_path=args.config)
    if not ctx.live:
        print(
            "Lab06 requires --live (and execution.live_api_enabled=true) to make "
            "real measurements against the Groq API; nothing was executed."
        )
        return

    model = args.model or ctx.config.groq.default_model
    all_results = []
    repetitions = args.repetitions or ctx.config.experiment.default_repetitions

    if args.experiment in ("non_streaming", "all"):
        print(f"Running lab06/non_streaming on {model} x{repetitions}...")
        for _ in range(repetitions):
            result = run_non_streaming(ctx.client, ctx.config, model)
            all_results.append(result)
            print(f"  client_elapsed_ms={result.performance.client_elapsed_ms:.1f} output_tokens_per_second={result.performance.output_tokens_per_second}")

    if args.experiment in ("streaming", "all"):
        print(f"Running lab06/streaming on {model} x{repetitions}...")
        for _ in range(repetitions):
            result = run_streaming(ctx.client, ctx.config, model)
            all_results.append(result)
            print(f"  ttft_ms={result.performance.ttft_ms} first_visible_content_ms={result.performance.first_visible_content_ms}")

    output_name = args.output_name or "lab06_streaming"
    write_json(all_results, ctx.output_dir / f"{output_name}.json")
    append_csv(all_results, ctx.output_dir / f"{output_name}.csv")
    print(f"Results written to {ctx.output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab06 - Streaming & Perceived Latency")
    parser.add_argument("experiment", choices=_EXPERIMENTS)
    parser.add_argument("--model", default=None)
    add_common_arguments(parser)
    parser.set_defaults(func=_run)
