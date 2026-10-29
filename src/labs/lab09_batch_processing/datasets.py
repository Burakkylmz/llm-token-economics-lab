"""Lab09's versioned golden dataset: synthetic, independently executable
business microtasks with deterministic expected answers.

Loaded once from data/lab09/golden_batch_dataset_v1.json and used
IDENTICALLY across sequential, concurrent, and Batch API execution -- the
only thing that should ever differ between conditions is the execution
mechanism, never the workload (spec Part 4, "Controlled Variables").
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "lab09"

EVALUATION_TYPE_EXACT_MATCH = "exact_match"
EVALUATION_TYPE_NUMERIC_MATCH = "numeric_match"
SUPPORTED_EVALUATION_TYPES = (EVALUATION_TYPE_EXACT_MATCH, EVALUATION_TYPE_NUMERIC_MATCH)

PROFILE_PILOT = "pilot"
PROFILE_DEFAULT = "default"
PROFILE_EXTENDED = "extended"
PROFILES = (PROFILE_PILOT, PROFILE_DEFAULT, PROFILE_EXTENDED)

# Task counts per spec Part 5's table. Pilot/default are stable PREFIXES of
# the full ordered task list (not a random subset), so a pilot run's tasks
# are always a subset of a default run's tasks, which are always a subset
# of extended's -- comparisons across profiles stay apples-to-apples.
PROFILE_TASK_COUNTS = {PROFILE_PILOT: 4, PROFILE_DEFAULT: 8, PROFILE_EXTENDED: 16}


class DatasetIntegrityError(ValueError):
    """Raised when the golden dataset file itself fails an offline validation check."""


@dataclass(frozen=True)
class GoldenTask:
    task_id: str
    category: str
    prompt: str
    expected_answer: str
    evaluation_type: str
    metadata: dict


def dataset_path(dataset_version: str = "v1") -> Path:
    return DATA_DIR / f"golden_batch_dataset_{dataset_version}.json"


def load_dataset(dataset_version: str = "v1") -> tuple[GoldenTask, ...]:
    path = dataset_path(dataset_version)
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    tasks = tuple(
        GoldenTask(
            task_id=t["task_id"],
            category=t["category"],
            prompt=t["prompt"],
            expected_answer=t["expected_answer"],
            evaluation_type=t["evaluation_type"],
            metadata=t.get("metadata", {}),
        )
        for t in raw["tasks"]
    )
    return tasks


def tasks_by_id(tasks: tuple[GoldenTask, ...]) -> dict[str, GoldenTask]:
    return {t.task_id: t for t in tasks}


def task_ids_for_profile(tasks: tuple[GoldenTask, ...], profile: str) -> tuple[str, ...]:
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile '{profile}'; must be one of {PROFILES}")
    count = PROFILE_TASK_COUNTS[profile]
    return tuple(t.task_id for t in tasks[:count])


def realtime_call_count_for_profile(profile: str) -> int:
    """Sequential + concurrent, one repetition each -- spec Part 5's table."""
    return 2 * PROFILE_TASK_COUNTS[profile]


def batch_request_count_for_profile(profile: str) -> int:
    """A Batch job with N input requests is NOT N real-time API calls."""
    return PROFILE_TASK_COUNTS[profile]


def check_dataset_integrity(tasks: tuple[GoldenTask, ...]) -> None:
    """Pure, offline validation -- raises DatasetIntegrityError on any violation.

    Checked here rather than assumed, since every downstream measurement
    (evaluation, pricing, reconciliation) silently trusts this dataset.
    """
    if len(tasks) < PROFILE_TASK_COUNTS[PROFILE_EXTENDED]:
        raise DatasetIntegrityError(
            f"Expected at least {PROFILE_TASK_COUNTS[PROFILE_EXTENDED]} tasks for the "
            f"extended profile, found {len(tasks)}"
        )

    seen_ids: set[str] = set()
    for task in tasks:
        if task.task_id in seen_ids:
            raise DatasetIntegrityError(f"Duplicate task_id '{task.task_id}'")
        seen_ids.add(task.task_id)

        if task.evaluation_type not in SUPPORTED_EVALUATION_TYPES:
            raise DatasetIntegrityError(
                f"Task '{task.task_id}' has unsupported evaluation_type "
                f"'{task.evaluation_type}'; must be one of {SUPPORTED_EVALUATION_TYPES}"
            )

        if not task.prompt.strip():
            raise DatasetIntegrityError(f"Task '{task.task_id}' has an empty prompt")

        if not task.expected_answer.strip():
            raise DatasetIntegrityError(f"Task '{task.task_id}' has an empty expected_answer")

        if task.evaluation_type == EVALUATION_TYPE_NUMERIC_MATCH:
            try:
                float(task.expected_answer)
            except ValueError as exc:
                raise DatasetIntegrityError(
                    f"Task '{task.task_id}' has evaluation_type=numeric_match but "
                    f"expected_answer '{task.expected_answer}' is not numeric"
                ) from exc

    for profile in PROFILES:
        if len(tasks) < PROFILE_TASK_COUNTS[profile]:
            raise DatasetIntegrityError(
                f"Profile '{profile}' needs {PROFILE_TASK_COUNTS[profile]} tasks, "
                f"dataset only has {len(tasks)}"
            )
