"""Root CLI entry point.

Each lab registers itself as a subcommand by appending a
(name, register_fn) pair to LAB_REGISTRY, where register_fn(subparsers)
adds that lab's argparse subparser. No lab is registered yet in Stage 0 —
this module is only the extension point.
"""

from __future__ import annotations

import argparse
from typing import Callable

LAB_REGISTRY: list[tuple[str, Callable[[argparse._SubParsersAction], None]]] = []


def build_parser() -> argparse.ArgumentParser:
    epilog = (
        "No labs are registered yet."
        if not LAB_REGISTRY
        else None
    )
    parser = argparse.ArgumentParser(
        prog="lab-cli",
        description="Token Economics, Cost & API Performance training labs (Groq API).",
        epilog=epilog,
    )
    subparsers = parser.add_subparsers(dest="lab", metavar="<lab>")

    for _name, register_fn in LAB_REGISTRY:
        register_fn(subparsers)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.lab is None:
        parser.print_help()


if __name__ == "__main__":
    main()
