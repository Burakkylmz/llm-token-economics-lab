from __future__ import annotations

import argparse
from pathlib import Path

from lab_core.config import load_config
from lab_core.results_writer import append_csv, write_json
from labs.lab07_rate_limiting.experiments import STRATEGIES, run_retry_strategy

LAB_ID = "lab07"


def _run(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print("Lab07 is fully simulated (no real Groq API calls, --live is ignored).")

    strategies = list(STRATEGIES) if args.experiment == "all" else [args.experiment]
    all_results = []

    for strategy in strategies:
        print(f"Running lab07/{strategy} with {args.fail_count} simulated 429s then success...")
        result = run_retry_strategy(
            strategy=strategy,
            retry_config=config.retry,
            fail_count=args.fail_count,
            retry_after_seconds=args.retry_after_seconds,
        )
        all_results.append(result)
        print(
            f"  succeeded={result.metadata.model_parameters['succeeded']} "
            f"retry_count={result.metadata.retry_count} "
            f"total_wait_seconds={result.metadata.model_parameters['total_wait_seconds']:.2f}"
        )

    output_name = args.output_name or "lab07_rate_limiting"
    output_dir = Path(config.experiment.results_dir)
    write_json(all_results, output_dir / f"{output_name}.json")
    append_csv(all_results, output_dir / f"{output_name}.csv")
    print(f"Results written to {output_dir}/{output_name}.json and .csv")


def register(subparsers: "argparse._SubParsersAction") -> None:
    parser = subparsers.add_parser(LAB_ID, help="Lab07 - Rate Limiting & Retry Strategies (fully simulated)")
    parser.add_argument("experiment", choices=[*STRATEGIES, "all"])
    parser.add_argument("--fail-count", type=int, default=2, dest="fail_count", help="Simulated 429s before success.")
    parser.add_argument(
        "--retry-after-seconds",
        type=str,
        default=None,
        help="If set, the simulated 429 responses carry this Retry-After header value.",
    )
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--live", action="store_true", help="Ignored: Lab07 never makes real API calls.")
    parser.add_argument("--output-name", type=str, default=None)
    parser.set_defaults(func=_run)
