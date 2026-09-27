"""
SENTINELAI — Step 59
Leakage-safe trajectory split for the provenance hard-negative dataset.

The split is performed at COMPLETE TRAJECTORY level.

Current dataset:
    400 trajectories
    6 records per trajectory
    2400 total records

Expected split:
    320 train trajectories
     80 test trajectories
    1920 train records
     480 test records

Both feature views must use the exact same trajectory partition.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42
TEST_SIZE = 0.20

BASE_DIR = Path("data/processed/hard_negative")

INPUT_WITHOUT = (
    BASE_DIR
    / "provenance_hard_negative_features_without_provenance.jsonl"
)

INPUT_WITH = (
    BASE_DIR
    / "provenance_hard_negative_features_with_provenance.jsonl"
)

OUTPUT_TRAIN_WITHOUT = (
    BASE_DIR
    / "train_without_provenance.jsonl"
)

OUTPUT_TEST_WITHOUT = (
    BASE_DIR
    / "test_without_provenance.jsonl"
)

OUTPUT_TRAIN_WITH = (
    BASE_DIR
    / "train_with_provenance.jsonl"
)

OUTPUT_TEST_WITH = (
    BASE_DIR
    / "test_with_provenance.jsonl"
)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:

        for line_number, line in enumerate(
            file,
            start=1,
        ):

            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)

            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}"
        )

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:

        for record in records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ---------------------------------------------------------------------------
# Basic helpers
# ---------------------------------------------------------------------------

def trajectory_id(
    record: dict[str, Any],
) -> str:

    value = record.get(
        "trajectory_id"
    )

    if not value:
        raise ValueError(
            "Record is missing trajectory_id."
        )

    return str(value)


def group_by_trajectory(
    records: list[dict[str, Any]],
) -> dict[
    str,
    list[dict[str, Any]]
]:

    grouped: dict[
        str,
        list[dict[str, Any]]
    ] = defaultdict(list)

    for record in records:

        grouped[
            trajectory_id(record)
        ].append(record)

    for trajectory in grouped.values():

        trajectory.sort(
            key=lambda record: int(
                record["step_number"]
            )
        )

    return dict(grouped)


# ---------------------------------------------------------------------------
# Structural validation
# ---------------------------------------------------------------------------

def validate_trajectory_structure(
    grouped: dict[
        str,
        list[dict[str, Any]]
    ],
) -> None:

    if not grouped:
        raise ValueError(
            "No trajectories found."
        )

    step_count_distribution = Counter()

    for trajectory, records in grouped.items():

        steps = [
            int(record["step_number"])
            for record in records
        ]

        step_count_distribution[
            len(steps)
        ] += 1

        expected = list(
            range(
                1,
                len(steps) + 1,
            )
        )

        if steps != expected:

            raise ValueError(
                f"Invalid step sequence for "
                f"trajectory {trajectory}: "
                f"{steps}. Expected {expected}."
            )

    print(
        "✓ Trajectory structure validated:"
    )

    for step_count, count in sorted(
        step_count_distribution.items()
    ):

        print(
            f"  {count} trajectories "
            f"with {step_count} steps"
        )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def trajectory_stratum(
    records: list[dict[str, Any]],
) -> str:

    if not records:
        raise ValueError(
            "Cannot determine stratum for empty trajectory."
        )

    first = records[0]

    source = str(
        first.get(
            "source_dataset",
            "unknown",
        )
    )

    variant = str(
        first.get(
            "variant",
            "unknown",
        )
    )

    return f"{source}|{variant}"


def build_split(
    grouped: dict[
        str,
        list[dict[str, Any]]
    ],
) -> tuple[list[str], list[str]]:

    trajectory_ids = sorted(
        grouped.keys()
    )

    strata = [
        trajectory_stratum(
            grouped[trajectory]
        )
        for trajectory in trajectory_ids
    ]

    stratum_counts = Counter(
        strata
    )

    print(
        "\nTrajectory strata:"
    )

    for stratum, count in sorted(
        stratum_counts.items()
    ):

        print(
            f"  {stratum}: {count}"
        )

    train_ids, test_ids = train_test_split(
        trajectory_ids,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=strata,
    )

    return (
        sorted(train_ids),
        sorted(test_ids),
    )


# ---------------------------------------------------------------------------
# Materialize split
# ---------------------------------------------------------------------------

def materialize(
    records: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    train_records: list[
        dict[str, Any]
    ] = []

    test_records: list[
        dict[str, Any]
    ] = []

    for record in records:

        current_id = trajectory_id(
            record
        )

        if current_id in train_ids:

            train_records.append(
                record
            )

        elif current_id in test_ids:

            test_records.append(
                record
            )

        else:

            raise ValueError(
                f"Trajectory {current_id} "
                "does not belong to train or test."
            )

    return (
        train_records,
        test_records,
    )


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------

def validate_no_overlap(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:

    train_ids = {
        trajectory_id(record)
        for record in train_records
    }

    test_ids = {
        trajectory_id(record)
        for record in test_records
    }

    overlap = (
        train_ids & test_ids
    )

    if overlap:

        raise ValueError(
            "TRAJECTORY LEAKAGE DETECTED!\n"
            f"Overlapping IDs: {sorted(overlap)}"
        )

    print(
        "✓ No trajectory appears in both "
        "train and test."
    )


def validate_same_partition(
    train_without: list[dict[str, Any]],
    test_without: list[dict[str, Any]],
    train_with: list[dict[str, Any]],
    test_with: list[dict[str, Any]],
) -> None:

    train_without_ids = {
        trajectory_id(record)
        for record in train_without
    }

    train_with_ids = {
        trajectory_id(record)
        for record in train_with
    }

    test_without_ids = {
        trajectory_id(record)
        for record in test_without
    }

    test_with_ids = {
        trajectory_id(record)
        for record in test_with
    }

    if train_without_ids != train_with_ids:

        raise ValueError(
            "Train partitions differ between "
            "provenance views."
        )

    if test_without_ids != test_with_ids:

        raise ValueError(
            "Test partitions differ between "
            "provenance views."
        )

    print(
        "✓ Both feature views use the exact "
        "same trajectory partition."
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    trajectories = {
        trajectory_id(record)
        for record in records
    }

    sources = Counter(
        str(
            record.get(
                "source_dataset",
                "unknown",
            )
        )
        for record in records
    )

    variants = Counter(
        str(
            record.get(
                "variant",
                "unknown",
            )
        )
        for record in records
    )

    labels = Counter(
        str(
            record.get(
                "label",
                "UNKNOWN",
            )
        )
        for record in records
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 60
    )

    print(
        f"Records:       {len(records)}"
    )

    print(
        f"Trajectories:  {len(trajectories)}"
    )

    print(
        "Sources:"
    )

    for source, count in sorted(
        sources.items()
    ):

        print(
            f"  {source}: {count}"
        )

    print(
        "Variants:"
    )

    for variant, count in sorted(
        variants.items()
    ):

        print(
            f"  {variant}: {count}"
        )

    print(
        "Labels:"
    )

    for label, count in sorted(
        labels.items()
    ):

        print(
            f"  {label}: {count}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "SENTINELAI — STEP 59"
    )
    print(
        "LEAKAGE-SAFE HARD-NEGATIVE TRAJECTORY SPLIT"
    )
    print("=" * 70)

    print(
        "\nLoading feature datasets..."
    )

    without_provenance = load_jsonl(
        INPUT_WITHOUT
    )

    with_provenance = load_jsonl(
        INPUT_WITH
    )

    print(
        f"Without-provenance records: "
        f"{len(without_provenance)}"
    )

    print(
        f"With-provenance records: "
        f"{len(with_provenance)}"
    )

    # ---------------------------------------------------------------
    # Alignment
    # ---------------------------------------------------------------

    without_keys = {
        (
            trajectory_id(record),
            int(record["step_number"]),
        )
        for record in without_provenance
    }

    with_keys = {
        (
            trajectory_id(record),
            int(record["step_number"]),
        )
        for record in with_provenance
    }

    if without_keys != with_keys:

        raise ValueError(
            "Feature views are not aligned."
        )

    print(
        "✓ Feature views are aligned."
    )

    # ---------------------------------------------------------------
    # Group trajectories
    # ---------------------------------------------------------------

    grouped = group_by_trajectory(
        without_provenance
    )

    grouped_with = group_by_trajectory(
        with_provenance
    )

    if set(grouped.keys()) != set(
        grouped_with.keys()
    ):

        raise ValueError(
            "Trajectory IDs differ between "
            "feature views."
        )

    validate_trajectory_structure(
        grouped
    )

    # ---------------------------------------------------------------
    # Build split
    # ---------------------------------------------------------------

    print(
        "\nBuilding trajectory-level split..."
    )

    train_ids, test_ids = build_split(
        grouped
    )

    train_id_set = set(
        train_ids
    )

    test_id_set = set(
        test_ids
    )

    print(
        f"\nTotal trajectories: "
        f"{len(train_ids) + len(test_ids)}"
    )

    print(
        f"Train trajectories: "
        f"{len(train_ids)}"
    )

    print(
        f"Test trajectories:  "
        f"{len(test_ids)}"
    )

    # ---------------------------------------------------------------
    # Materialize
    # ---------------------------------------------------------------

    (
        train_without,
        test_without,
    ) = materialize(
        without_provenance,
        train_id_set,
        test_id_set,
    )

    (
        train_with,
        test_with,
    ) = materialize(
        with_provenance,
        train_id_set,
        test_id_set,
    )

    # ---------------------------------------------------------------
    # Leakage checks
    # ---------------------------------------------------------------

    validate_no_overlap(
        train_without,
        test_without,
    )

    validate_same_partition(
        train_without,
        test_without,
        train_with,
        test_with,
    )

    # ---------------------------------------------------------------
    # Expected dataset size
    # ---------------------------------------------------------------

    expected_trajectories = 400
    expected_train_trajectories = 320
    expected_test_trajectories = 80
    expected_steps = 6

    actual_trajectories = (
        len(train_ids)
        + len(test_ids)
    )

    if actual_trajectories != (
        expected_trajectories
    ):

        raise ValueError(
            f"Expected {expected_trajectories} "
            f"trajectories but found "
            f"{actual_trajectories}."
        )

    if len(train_ids) != (
        expected_train_trajectories
    ):

        raise ValueError(
            f"Expected {expected_train_trajectories} "
            f"train trajectories but found "
            f"{len(train_ids)}."
        )

    if len(test_ids) != (
        expected_test_trajectories
    ):

        raise ValueError(
            f"Expected {expected_test_trajectories} "
            f"test trajectories but found "
            f"{len(test_ids)}."
        )

    expected_train_records = (
        expected_train_trajectories
        * expected_steps
    )

    expected_test_records = (
        expected_test_trajectories
        * expected_steps
    )

    if len(train_without) != (
        expected_train_records
    ):

        raise ValueError(
            f"Expected {expected_train_records} "
            f"train records but found "
            f"{len(train_without)}."
        )

    if len(test_without) != (
        expected_test_records
    ):

        raise ValueError(
            f"Expected {expected_test_records} "
            f"test records but found "
            f"{len(test_without)}."
        )

    # ---------------------------------------------------------------
    # Save
    # ---------------------------------------------------------------

    write_jsonl(
        OUTPUT_TRAIN_WITHOUT,
        train_without,
    )

    write_jsonl(
        OUTPUT_TEST_WITHOUT,
        test_without,
    )

    write_jsonl(
        OUTPUT_TRAIN_WITH,
        train_with,
    )

    write_jsonl(
        OUTPUT_TEST_WITH,
        test_with,
    )

    # ---------------------------------------------------------------
    # Statistics
    # ---------------------------------------------------------------

    print_statistics(
        "TRAIN — WITHOUT PROVENANCE",
        train_without,
    )

    print_statistics(
        "TEST — WITHOUT PROVENANCE",
        test_without,
    )

    print_statistics(
        "TRAIN — WITH PROVENANCE",
        train_with,
    )

    print_statistics(
        "TEST — WITH PROVENANCE",
        test_with,
    )

    # ---------------------------------------------------------------
    # Output
    # ---------------------------------------------------------------

    print(
        "\nOutput files:"
    )

    print(
        f"  ✓ {OUTPUT_TRAIN_WITHOUT}"
    )

    print(
        f"  ✓ {OUTPUT_TEST_WITHOUT}"
    )

    print(
        f"  ✓ {OUTPUT_TRAIN_WITH}"
    )

    print(
        f"  ✓ {OUTPUT_TEST_WITH}"
    )

    print("\n" + "=" * 70)
    print(
        "✓ STEP 59 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        "\nThe hard-negative dataset was split by "
        "complete trajectory with no train/test "
        "trajectory overlap."
    )


if __name__ == "__main__":
    main()