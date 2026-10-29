"""Root CLI entry point.

Each lab module exposes a `register(subparsers)` function that adds its
own argparse subparser and calls `parser.set_defaults(func=<its run fn>)`.
This module only discovers and wires those together; it contains no
lab-specific logic itself.
"""

from __future__ import annotations

import argparse
from typing import Callable

LabRegisterFn = Callable[["argparse._SubParsersAction"], None]

# Populated by _discover_labs(). Kept as a module-level list (rather than
# computed fresh each call) so tests can inspect/override it directly.
LAB_REGISTRY: list[tuple[str, LabRegisterFn]] = []


def _discover_labs() -> list[tuple[str, LabRegisterFn]]:
    """Import every lab package and collect its register() function.

    A lab is "discovered" by explicit name here (not by scanning the
    filesystem) so that a half-written lab directory can never be picked
    up accidentally.
    """
    import importlib

    lab_modules = [
        "labs.lab01_token_anatomy.cli",
        "labs.lab02_context_growth.cli",
        "labs.lab03_prompt_caching.cli",
        "labs.lab04_output_control.cli",
        "labs.lab05_model_routing.cli",
        "labs.lab06_streaming.cli",
        "labs.lab07_rate_limiting.cli",
        "labs.lab08_concurrency.cli",
        "labs.lab09_batch_processing.cli",
        "labs.lab10_conversation_summarization.cli",
    ]

    registry: list[tuple[str, LabRegisterFn]] = []
    for module_path in lab_modules:
        module = importlib.import_module(module_path)
        registry.append((module.LAB_ID, module.register))
    return registry


def build_parser() -> argparse.ArgumentParser:
    global LAB_REGISTRY
    if not LAB_REGISTRY:
        LAB_REGISTRY = _discover_labs()

    parser = argparse.ArgumentParser(
        prog="lab-cli",
        description="Token Economics, Cost & API Performance training labs (Groq API).",
    )
    subparsers = parser.add_subparsers(dest="lab", metavar="<lab>")

    for _lab_id, register_fn in LAB_REGISTRY:
        register_fn(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if getattr(args, "func", None) is None:
        parser.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
