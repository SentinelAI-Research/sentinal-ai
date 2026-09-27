from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_experiment.jsonl"
)

EXPECTED_TRAJECTORIES = 100
EXPECTED_STEPS = 6

EXPECTED_TOOLS = {
    1: "read_file",
    2: "summarize",
    3: "write_memory",
    4: "read_memory",
    5: "paraphrase",
    6: "send_message",
}

EXPECTED_LABELS = {
    1: "SAFE",
    2: "EARLY_RISK",
    3: "EARLY_RISK",
    4: "EARLY_RISK",
    5: "EARLY_RISK",
    6: "VIOLATION",
}

EXPECTED_TRANSFORMATIONS = {
    1: "source_ingestion",
    2: "summarize",
    3: "write_memory",
    4: "read_memory",
    5: "paraphrase",
    6: "send_message",
}


def fail(message: str) -> None:
    raise RuntimeError(f"VALIDATION FAILED: {message}")


def main() -> None:
    print("=" * 70)
    print("SENTINELAI — ENRON EXPERIMENT DATASET VALIDATION")
    print("=" * 70)
    print(f"Dataset: {DATASET_PATH}")
    print()

    if not DATASET_PATH.exists():
        fail(f"Dataset does not exist: {DATASET_PATH}")

    records = []

    with DATASET_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(
                    f"Invalid JSON at line {line_number}: {exc}"
                )

            if not isinstance(record, dict):
                fail(
                    f"Line {line_number} does not contain a JSON object."
                )

            records.append(record)

    print(f"Total records loaded: {len(records)}")

    expected_total = EXPECTED_TRAJECTORIES * EXPECTED_STEPS

    if len(records) != expected_total:
        fail(
            f"Expected {expected_total} records, "
            f"found {len(records)}."
        )

    print(f"✓ Record count = {expected_total}")

    required_fields = {
        "trajectory_id",
        "scenario_id",
        "scenario_type",
        "step_number",
        "total_steps",
        "tool_name",
        "input_object_ids",
        "output_object_id",
        "label",
        "provenance_depth",
        "transformation_depth",
        "source",
        "sensitivity",
        "transformation",
        "content_hash",
    }

    for index, record in enumerate(records, start=1):
        missing = required_fields - set(record.keys())

        if missing:
            fail(
                f"Record {index} is missing fields: "
                f"{sorted(missing)}"
            )

    print("✓ Required fields present")

    trajectory_ids = {
        record["trajectory_id"]
        for record in records
    }

    if len(trajectory_ids) != EXPECTED_TRAJECTORIES:
        fail(
            f"Expected {EXPECTED_TRAJECTORIES} unique trajectories, "
            f"found {len(trajectory_ids)}."
        )

    print(
        f"✓ Unique trajectories = {EXPECTED_TRAJECTORIES}"
    )

    trajectory_records: dict[str, list[dict]] = {}

    for record in records:
        trajectory_id = record["trajectory_id"]

        trajectory_records.setdefault(
            trajectory_id,
            []
        ).append(record)

    for trajectory_id, trajectory in trajectory_records.items():

        if len(trajectory) != EXPECTED_STEPS:
            fail(
                f"Trajectory {trajectory_id} contains "
                f"{len(trajectory)} records instead of "
                f"{EXPECTED_STEPS}."
            )

        trajectory.sort(
            key=lambda record: record["step_number"]
        )

        expected_steps = list(range(1, EXPECTED_STEPS + 1))
        actual_steps = [
            record["step_number"]
            for record in trajectory
        ]

        if actual_steps != expected_steps:
            fail(
                f"Trajectory {trajectory_id} has incorrect "
                f"step sequence: {actual_steps}"
            )

        for record in trajectory:

            step = record["step_number"]

            if record["total_steps"] != EXPECTED_STEPS:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"total_steps should be {EXPECTED_STEPS}, "
                    f"found {record['total_steps']}."
                )

            expected_tool = EXPECTED_TOOLS[step]

            if record["tool_name"] != expected_tool:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected tool '{expected_tool}', "
                    f"found '{record['tool_name']}'."
                )

            expected_label = EXPECTED_LABELS[step]

            if record["label"] != expected_label:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected label '{expected_label}', "
                    f"found '{record['label']}'."
                )

            expected_transformation = EXPECTED_TRANSFORMATIONS[step]

            if record["transformation"] != expected_transformation:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected transformation "
                    f"'{expected_transformation}', "
                    f"found '{record['transformation']}'."
                )

            expected_provenance_depth = step - 1

            if record["provenance_depth"] != expected_provenance_depth:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected provenance depth "
                    f"{expected_provenance_depth}, "
                    f"found {record['provenance_depth']}."
                )

            expected_transformation_depth = step

            if record["transformation_depth"] != expected_transformation_depth:
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected transformation depth "
                    f"{expected_transformation_depth}, "
                    f"found {record['transformation_depth']}."
                )

            if record["sensitivity"] != "PUBLIC":
                fail(
                    f"{trajectory_id} step {step}: "
                    f"expected sensitivity PUBLIC, "
                    f"found {record['sensitivity']}."
                )

            if not record["output_object_id"]:
                fail(
                    f"{trajectory_id} step {step}: "
                    "output_object_id is empty."
                )

            if not record["content_hash"]:
                fail(
                    f"{trajectory_id} step {step}: "
                    "content_hash is empty."
                )

            if not isinstance(record["input_object_ids"], list):
                fail(
                    f"{trajectory_id} step {step}: "
                    "input_object_ids must be a list."
                )

        # ---------------------------------------------------------
        # Verify provenance chain
        # ---------------------------------------------------------

        first = trajectory[0]

        if first["input_object_ids"] != []:
            fail(
                f"{trajectory_id} step 1 should have no parents."
            )

        for index in range(1, EXPECTED_STEPS):

            previous_record = trajectory[index - 1]
            current_record = trajectory[index]

            expected_parent = previous_record["output_object_id"]
            actual_parents = current_record["input_object_ids"]

            if actual_parents != [expected_parent]:
                fail(
                    f"{trajectory_id} step "
                    f"{current_record['step_number']}: "
                    f"expected parent "
                    f"[{expected_parent!r}], "
                    f"found {actual_parents!r}."
                )

        # ---------------------------------------------------------
        # Verify source consistency
        # ---------------------------------------------------------

        sources = {
            record["source"]
            for record in trajectory
        }

        if len(sources) != 1:
            fail(
                f"{trajectory_id} has multiple source values: "
                f"{sources}"
            )

    print("✓ Every trajectory contains exactly 6 ordered steps")
    print("✓ Tool sequence validated")
    print("✓ Label sequence validated")
    print("✓ Transformation sequence validated")
    print("✓ Provenance depth validated")
    print("✓ Transformation depth validated")
    print("✓ Sensitivity values validated")
    print("✓ Content hashes are present")
    print("✓ Parent-child provenance chains validated")
    print("✓ Source consistency validated")

    # -------------------------------------------------------------
    # Global duplicate check
    # -------------------------------------------------------------

    trajectory_step_pairs = [
        (
            record["trajectory_id"],
            record["step_number"],
        )
        for record in records
    ]

    duplicate_pairs = [
        pair
        for pair, count in Counter(trajectory_step_pairs).items()
        if count > 1
    ]

    if duplicate_pairs:
        fail(
            f"Duplicate trajectory/step pairs found: "
            f"{duplicate_pairs[:10]}"
        )

    print("✓ No duplicate trajectory/step pairs")

    # -------------------------------------------------------------
    # Global label distribution
    # -------------------------------------------------------------

    label_counts = Counter(
        record["label"]
        for record in records
    )

    expected_label_counts = {
        "SAFE": EXPECTED_TRAJECTORIES,
        "EARLY_RISK": EXPECTED_TRAJECTORIES * 4,
        "VIOLATION": EXPECTED_TRAJECTORIES,
    }

    if dict(label_counts) != expected_label_counts:
        fail(
            "Unexpected label distribution. "
            f"Expected {expected_label_counts}, "
            f"found {dict(label_counts)}."
        )

    print(
        "✓ Label distribution:",
        dict(label_counts)
    )

    # -------------------------------------------------------------
    # Global scenario consistency
    # -------------------------------------------------------------

    scenario_ids = {
        record["scenario_id"]
        for record in records
    }

    scenario_types = {
        record["scenario_type"]
        for record in records
    }

    if scenario_ids != {
        "enron-memory-mediated-leakage"
    }:
        fail(
            f"Unexpected scenario IDs: {scenario_ids}"
        )

    if scenario_types != {
        "LONG_HORIZON_LEAKAGE"
    }:
        fail(
            f"Unexpected scenario types: {scenario_types}"
        )

    print("✓ Scenario metadata validated")

    print()
    print("=" * 70)
    print("VALIDATION PASSED")
    print("=" * 70)
    print(
        f"Validated {len(records)} records "
        f"across {len(trajectory_ids)} trajectories."
    )
    print(
        "Dataset is structurally consistent and ready "
        "for the next ML preparation phase."
    )


if __name__ == "__main__":
    main()