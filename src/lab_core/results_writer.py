"""Writers that persist ExperimentResult records as JSON and CSV.

JSON preserves the full nested structure; CSV flattens each result into a
single row (prefixing fields with their section name) so results from
multiple runs can be compared side by side in a spreadsheet.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

from lab_core.metrics import ExperimentResult


def _flatten(result: ExperimentResult) -> dict:
    flat: dict = {}
    as_dict = asdict(result)
    for section_name, section_values in as_dict.items():
        if isinstance(section_values, dict):
            for key, value in section_values.items():
                flat[f"{section_name}.{key}"] = value
        else:
            flat[section_name] = section_values
    return flat


def write_json(results: list[ExperimentResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(result) for result in results]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def append_csv(results: list[ExperimentResult], output_path: Path) -> None:
    if not results:
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [_flatten(result) for result in results]
    fieldnames = list(rows[0].keys())

    file_exists = output_path.exists()
    with open(output_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
