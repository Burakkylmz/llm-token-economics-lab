"""Lab05's hand-crafted golden dataset: 13 deterministic, auditable tasks
spanning 8 required categories (plus one bonus code-generation task),
loaded from `golden_dataset.json` next to this module.

Expected answers are deterministic and the dataset's `complexity_label` is
GROUND-TRUTH METADATA that a real deployment-time router would never have
access to -- it exists for offline/instructor analysis only. See
`labs.lab05_model_routing.routing` for the hard rule that the router may
only ever read a task's `prompt` text, never this label, never
`expected_answer`, never any evaluation result.

`PILOT_TASK_IDS` (4) and `DEFAULT_TASK_IDS` (6) are deliberately
PREDETERMINED, stratified subsets -- not a random sample -- so every run
of the same profile selects the exact same tasks (reproducibility), while
still covering distinct categories (stratification). The remaining tasks
in the full dataset (including the bonus code-generation task) are
available for `--profile extended`'s broader instructor discussion and for
offline unit testing of every evaluator/routing-rule combination, but are
never part of a live Grand Test's planned calls unless explicitly selected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

_DATASET_PATH = Path(__file__).resolve().parent / "golden_dataset.json"


@dataclass(frozen=True)
class GoldenTask:
    task_id: str
    category: str
    complexity_label: str
    prompt: str
    expected_answer: str
    evaluation_type: str
    evaluation_config: dict
    source: str
    notes: str = ""


def _load_dataset() -> tuple[GoldenTask, ...]:
    with open(_DATASET_PATH, encoding="utf-8") as f:
        raw = json.load(f)
    return tuple(
        GoldenTask(
            task_id=r["task_id"],
            category=r["category"],
            complexity_label=r["complexity_label"],
            prompt=r["prompt"],
            expected_answer=r["expected_answer"],
            evaluation_type=r["evaluation_type"],
            evaluation_config=r.get("evaluation_config", {}),
            source=r.get("source", "handcrafted"),
            notes=r.get("notes", ""),
        )
        for r in raw
    )


GOLDEN_DATASET: tuple[GoldenTask, ...] = _load_dataset()
TASK_BY_ID: dict[str, GoldenTask] = {t.task_id: t for t in GOLDEN_DATASET}

# Predetermined, stratified, reproducible task selections -- see module
# docstring. Both are checked (in tests) to reference only real task_ids
# and to span more than one category.
PILOT_TASK_IDS: tuple[str, ...] = ("task_001", "task_003", "task_007", "task_010")
DEFAULT_TASK_IDS: tuple[str, ...] = ("task_001", "task_003", "task_005", "task_007", "task_009", "task_011")


def task_ids_for_profile(profile: str) -> tuple[str, ...]:
    """"pilot" -> PILOT_TASK_IDS; "default"/"extended" -> DEFAULT_TASK_IDS
    (extended keeps the same 6 tasks and only adds the always_large
    strategy -- see labs.lab05_model_routing.matrix)."""
    if profile == "pilot":
        return PILOT_TASK_IDS
    return DEFAULT_TASK_IDS


def get_task(task_id: str) -> Optional[GoldenTask]:
    return TASK_BY_ID.get(task_id)
