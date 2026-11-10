"""Shared CLI plumbing reused by every lab's cli.py.

Centralizes the safe-execution guard (the --live flag combined with
config.execution.live_api_enabled) and config/client construction, so this
policy is enforced identically everywhere instead of being reimplemented
once per lab.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from groq import AsyncGroq, Groq

from lab_core.config import AppConfig, load_config


class LiveExecutionNotAllowed(RuntimeError):
    """Raised when --live was requested but execution is not permitted."""


@dataclass
class LabContext:
    config: AppConfig
    live: bool
    output_dir: Path
    client: Optional[Groq]
    async_client: Optional[AsyncGroq]


def build_context(requested_live: bool, config_path: Optional[Path] = None) -> LabContext:
    """Resolve configuration and (if allowed) construct real Groq clients.

    Raises LiveExecutionNotAllowed rather than silently downgrading to a
    simulated run, so a participant who explicitly asked for --live always
    gets an explicit reason when real execution does not happen.
    """
    config = load_config(config_path)

    if requested_live and not config.execution.live_api_enabled:
        raise LiveExecutionNotAllowed(
            "Live execution requested (--live) but execution.live_api_enabled "
            "is false in configuration. Set it to true in your config to allow "
            "real Groq API calls."
        )

    live = requested_live and config.execution.live_api_enabled
    client: Optional[Groq] = None
    async_client: Optional[AsyncGroq] = None

    if live:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise LiveExecutionNotAllowed(
                "Live execution requested but GROQ_API_KEY is not set in the environment."
            )
        print(f"[COST WARNING] {config.execution.cost_warning_message}")
        # One retry owner: our experiment runners count and time retries explicitly.
        # Disable SDK retries so they cannot silently multiply HTTP attempts.
        client_options = {
            "api_key": api_key,
            "timeout": config.groq.request_timeout_seconds,
            "max_retries": 0,
        }
        client = Groq(**client_options)
        async_client = AsyncGroq(**client_options)

    return LabContext(
        config=config,
        live=live,
        output_dir=Path(config.experiment.results_dir),
        client=client,
        async_client=async_client,
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--live",
        action="store_true",
        help="Make real Groq API calls (may incur real cost on your account).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to a TOML config file (default: config/default.toml).",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=None,
        help="Override the number of repetitions per experiment condition.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default=None,
        help="Base filename (without extension) for the JSON/CSV results written to results_dir.",
    )
