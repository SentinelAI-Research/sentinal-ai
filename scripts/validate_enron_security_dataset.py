from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_features.jsonl"
)

REQUIRED_FIELDS = {
    "trajectory_id",
    "scenario_id",
    "step_number",
    "total_steps",
    "trajectory_progress",
    "provenance_depth",
    "transformation_depth",
    "tool_name",
    "tool_encoded",
    "is_memory_operation",
    "is_external_action",
    "is_transformation",
    "has_input_object",
    "sensitivity",
    "transformation",
    "label",
    "label_encoded",
}

VALID_LABELS = {
    "SAFE",
    "EARLY_RISK",
    "VIOLATION",
}


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {INPUT_PATH}"
        )

    records = []

    with INPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        raise ValueError(
            "Dataset is empty."
        )

    seen = set()

    for record in records:

        missing = REQUIRED_FIELDS - set(record)

        if missing:
            raise ValueError(
                f"Missing fields: {sorted(missing)}"
            )

        key = (
            record["trajectory_id"],
            record["step_number"],
        )

        if key in seen:
            raise ValueError(
                f"Duplicate trajectory step: {key}"
            )

        seen.add(key)

        if record["label"] not in VALID_LABELS:
            raise ValueError(
                f"Invalid label: {record['label']}"
            )

        if record["total_steps"] != 6:
            raise ValueError(
                "Every Enron trajectory must contain 6 steps."
            )

    trajectory_ids = sorted(
        {
            record["trajectory_id"]
            for record in records
        }
    )

    for trajectory_id in trajectory_ids:

        trajectory = [
            record
            for record in records
            if record["trajectory_id"] == trajectory_id
        ]

        trajectory.sort(
            key=lambda record: record["step_number"]
        )

        expected_steps = list(range(1, 7))

        actual_steps = [
            record["step_number"]
            for record in trajectory
        ]

        if actual_steps != expected_steps:
            raise ValueError(
                f"Invalid steps for {trajectory_id}: "
                f"{actual_steps}"
            )

        expected_provenance = list(range(0, 6))
        actual_provenance = [
            record["provenance_depth"]
            for record in trajectory
        ]

        if actual_provenance != expected_provenance:
            raise ValueError(
                f"Invalid provenance depths for "
                f"{trajectory_id}: "
                f"{actual_provenance}"
            )

        expected_transformation = list(range(1, 7))
        actual_transformation = [
            record["transformation_depth"]
            for record in trajectory
        ]

        if actual_transformation != expected_transformation:
            raise ValueError(
                f"Invalid transformation depths for "
                f"{trajectory_id}: "
                f"{actual_transformation}"
            )

    print("ENRON DATASET VALIDATION PASSED")
    print(f"Total records: {len(records)}")
    print(f"Total trajectories: {len(trajectory_ids)}")
    print("Duplicate trajectory steps: 0")
    print("Missing required fields: 0")
    print("Invalid labels: 0")
    print("Invalid provenance depths: 0")
    print("Invalid transformation depths: 0")


if __name__ == "__main__":
    main()