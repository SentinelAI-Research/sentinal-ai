"""
SENTINELAI — Leakage-Safe Trajectory Split

Splits the combined security ML dataset by COMPLETE TRAJECTORY.

Important:
- Never split individual rows from the same trajectory.
- The same trajectory must never appear in both train and test.
- Both provenance views must use exactly the same trajectory partition.
- AMI and Enron trajectories are stratified so both datasets are represented
  in train and test.
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

BASE_DIR = Path("data/processed/combined")

INPUT_WITHOUT_PROVENANCE = (
    BASE_DIR / "security_ml_features_without_provenance.jsonl"
)

INPUT_WITH_PROVENANCE = (
    BASE_DIR / "security_ml_features_with_provenance.jsonl"
)

OUTPUT_TRAIN_WITHOUT_PROVENANCE = (
    BASE_DIR / "train_without_provenance.jsonl"
)

OUTPUT_TEST_WITHOUT_PROVENANCE = (
    BASE_DIR / "test_without_provenance.jsonl"
)

OUTPUT_TRAIN_WITH_PROVENANCE = (
    BASE_DIR / "train_with_provenance.jsonl"
)

OUTPUT_TEST_WITH_PROVENANCE = (
    BASE_DIR / "test_with_provenance.jsonl"
)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            records.append(record)

    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def get_trajectory_id(record: dict[str, Any]) -> str:
    trajectory_id = record.get("trajectory_id")

    if not trajectory_id:
        raise ValueError(
            "Record is missing required 'trajectory_id': "
            f"{record}"
        )

    return str(trajectory_id)


def get_source_dataset(record: dict[str, Any]) -> str:
    """
    source_dataset was added during the combined-dataset stage.

    If it is missing, derive it from the trajectory ID as a fallback.
    This keeps the splitter robust without changing the source dataset.
    """

    source_dataset = record.get("source_dataset")

    if source_dataset:
        return str(source_dataset)

    trajectory_id = get_trajectory_id(record).lower()

    if trajectory_id.startswith("ami"):
        return "ami"

    if trajectory_id.startswith("enron"):
        return "enron"

    raise ValueError(
        "Unable to determine source dataset for trajectory "
        f"{trajectory_id}. Expected 'source_dataset' field or "
        "an AMI/Enron trajectory ID."
    )


def get_scenario(record: dict[str, Any]) -> str:
    """
    Use scenario_id if available.

    The current combined feature files do not necessarily contain
    scenario_type, so we intentionally do NOT require that field.
    """

    scenario_id = record.get("scenario_id")

    if scenario_id:
        return str(scenario_id)

    # Current controlled dataset uses one scenario per source dataset.
    # Therefore source_dataset is a safe fallback stratum.
    return get_source_dataset(record)


# ---------------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------------

def validate_feature_view_alignment(
    without_provenance: list[dict[str, Any]],
    with_provenance: list[dict[str, Any]],
) -> None:

    without_keys = {
        (
            get_trajectory_id(record),
            int(record.get("step_number", -1)),
        )
        for record in without_provenance
    }

    with_keys = {
        (
            get_trajectory_id(record),
            int(record.get("step_number", -1)),
        )
        for record in with_provenance
    }

    if without_keys != with_keys:
        only_without = without_keys - with_keys
        only_with = with_keys - without_keys

        raise ValueError(
            "Feature views are not aligned.\n"
            f"Only in without-provenance: {len(only_without)}\n"
            f"Only in with-provenance: {len(only_with)}"
        )

    print("✓ Feature views are aligned.")


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------

def group_records_by_trajectory(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        trajectory_id = get_trajectory_id(record)
        grouped[trajectory_id].append(record)

    return dict(grouped)


def validate_trajectory_structure(
    grouped: dict[str, list[dict[str, Any]]],
) -> None:

    if not grouped:
        raise ValueError("No trajectories found.")

    step_counts = Counter()

    for trajectory_id, records in grouped.items():
        steps = sorted(
            int(record["step_number"])
            for record in records
            if "step_number" in record
        )

        if not steps:
            raise ValueError(
                f"Trajectory {trajectory_id} has no step_number values."
            )

        step_counts[len(steps)] += 1

        expected_steps = list(range(1, len(steps) + 1))

        if steps != expected_steps:
            raise ValueError(
                f"Trajectory {trajectory_id} has invalid step sequence: "
                f"{steps}. Expected: {expected_steps}"
            )

    print(
        f"✓ Validated {len(grouped)} trajectories "
        f"with step structure: {dict(step_counts)}"
    )


# ---------------------------------------------------------------------------
# Stratification
# ---------------------------------------------------------------------------

def trajectory_stratum(
    records: list[dict[str, Any]],
) -> str:

    if not records:
        raise ValueError("Cannot create stratum for empty trajectory.")

    first = records[0]

    source_dataset = get_source_dataset(first)
    scenario = get_scenario(first)

    return f"{source_dataset}|{scenario}"


def build_group_split(
    grouped: dict[str, list[dict[str, Any]]],
) -> tuple[list[str], list[str]]:

    trajectory_ids = sorted(grouped.keys())

    strata = [
        trajectory_stratum(grouped[trajectory_id])
        for trajectory_id in trajectory_ids
    ]

    stratum_counts = Counter(strata)

    print("\nTrajectory strata:")
    for stratum, count in sorted(stratum_counts.items()):
        print(f"  {stratum}: {count}")

    if len(trajectory_ids) < 2:
        raise ValueError("At least two trajectories are required.")

    try:
        train_ids, test_ids = train_test_split(
            trajectory_ids,
            test_size=TEST_SIZE,
            random_state=RANDOM_SEED,
            stratify=strata,
        )

    except ValueError as exc:
        raise ValueError(
            "Unable to perform stratified trajectory split. "
            f"Original error: {exc}"
        ) from exc

    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)

    return train_ids, test_ids


# ---------------------------------------------------------------------------
# Split materialization
# ---------------------------------------------------------------------------

def materialize_split(
    records: list[dict[str, Any]],
    train_ids: set[str],
    test_ids: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:

    train_records: list[dict[str, Any]] = []
    test_records: list[dict[str, Any]] = []

    seen_ids: set[str] = set()

    for record in records:
        trajectory_id = get_trajectory_id(record)

        if trajectory_id in seen_ids:
            # Multiple rows per trajectory are expected.
            pass

        if trajectory_id in train_ids:
            train_records.append(record)

        elif trajectory_id in test_ids:
            test_records.append(record)

        else:
            raise ValueError(
                f"Trajectory {trajectory_id} belongs to neither "
                "train nor test."
            )

    return train_records, test_records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_split_statistics(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    trajectory_ids = {
        get_trajectory_id(record)
        for record in records
    }

    source_counts = Counter(
        get_source_dataset(record)
        for record in records
    )

    label_counts = Counter(
        record.get("label", "UNKNOWN")
        for record in records
    )

    print(f"\n{name}")
    print("-" * 60)
    print(f"Records:       {len(records)}")
    print(f"Trajectories:  {len(trajectory_ids)}")

    print("Source datasets:")
    for source, count in sorted(source_counts.items()):
        print(f"  {source}: {count}")

    print("Labels:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")


# ---------------------------------------------------------------------------
# Leakage checks
# ---------------------------------------------------------------------------

def validate_no_trajectory_leakage(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:

    train_ids = {
        get_trajectory_id(record)
        for record in train_records
    }

    test_ids = {
        get_trajectory_id(record)
        for record in test_records
    }

    overlap = train_ids & test_ids

    if overlap:
        raise ValueError(
            "TRAJECTORY LEAKAGE DETECTED!\n"
            f"Overlapping trajectories: {sorted(overlap)}"
        )

    print("✓ No trajectory appears in both train and test.")


def validate_same_partition(
    train_without: list[dict[str, Any]],
    test_without: list[dict[str, Any]],
    train_with: list[dict[str, Any]],
    test_with: list[dict[str, Any]],
) -> None:

    train_without_ids = {
        get_trajectory_id(record)
        for record in train_without
    }

    train_with_ids = {
        get_trajectory_id(record)
        for record in train_with
    }

    test_without_ids = {
        get_trajectory_id(record)
        for record in test_without
    }

    test_with_ids = {
        get_trajectory_id(record)
        for record in test_with
    }

    if train_without_ids != train_with_ids:
        raise ValueError(
            "Train trajectory partitions differ between feature views."
        )

    if test_without_ids != test_with_ids:
        raise ValueError(
            "Test trajectory partitions differ between feature views."
        )

    print("✓ Both feature views use the identical trajectory partition.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("SENTINELAI — LEAKAGE-SAFE TRAJECTORY SPLIT")
    print("=" * 70)

    without_provenance = load_jsonl(
        INPUT_WITHOUT_PROVENANCE
    )

    with_provenance = load_jsonl(
        INPUT_WITH_PROVENANCE
    )

    print(
        f"\nLoaded without-provenance records: "
        f"{len(without_provenance)}"
    )

    print(
        f"Loaded with-provenance records: "
        f"{len(with_provenance)}"
    )

    validate_feature_view_alignment(
        without_provenance,
        with_provenance,
    )

    grouped_without = group_records_by_trajectory(
        without_provenance
    )

    grouped_with = group_records_by_trajectory(
        with_provenance
    )

    if set(grouped_without) != set(grouped_with):
        raise ValueError(
            "Trajectory IDs differ between the two feature views."
        )

    validate_trajectory_structure(grouped_without)

    print("\nBuilding trajectory-level split...")

    train_ids, test_ids = build_group_split(
        grouped_without
    )

    train_id_set = set(train_ids)
    test_id_set = set(test_ids)

    print(f"\nTotal trajectories: {len(train_ids) + len(test_ids)}")
    print(f"Train trajectories: {len(train_ids)}")
    print(f"Test trajectories:  {len(test_ids)}")

    train_without, test_without = materialize_split(
        without_provenance,
        train_id_set,
        test_id_set,
    )

    train_with, test_with = materialize_split(
        with_provenance,
        train_id_set,
        test_id_set,
    )

    validate_no_trajectory_leakage(
        train_without,
        test_without,
    )

    validate_same_partition(
        train_without,
        test_without,
        train_with,
        test_with,
    )

    # Expected sizes for current dataset:
    # 200 trajectories × 6 steps = 1200 records
    expected_total_trajectories = 200
    expected_train_trajectories = 160
    expected_test_trajectories = 40
    expected_steps_per_trajectory = 6

    total_trajectories = len(train_ids) + len(test_ids)

    if total_trajectories != expected_total_trajectories:
        raise ValueError(
            f"Expected {expected_total_trajectories} trajectories, "
            f"found {total_trajectories}."
        )

    if len(train_ids) != expected_train_trajectories:
        raise ValueError(
            f"Expected {expected_train_trajectories} train trajectories, "
            f"found {len(train_ids)}."
        )

    if len(test_ids) != expected_test_trajectories:
        raise ValueError(
            f"Expected {expected_test_trajectories} test trajectories, "
            f"found {len(test_ids)}."
        )

    expected_train_records = (
        expected_train_trajectories * expected_steps_per_trajectory
    )

    expected_test_records = (
        expected_test_trajectories * expected_steps_per_trajectory
    )

    if len(train_without) != expected_train_records:
        raise ValueError(
            f"Expected {expected_train_records} train records, "
            f"found {len(train_without)}."
        )

    if len(test_without) != expected_test_records:
        raise ValueError(
            f"Expected {expected_test_records} test records, "
            f"found {len(test_without)}."
        )

    # Write outputs.
    write_jsonl(
        OUTPUT_TRAIN_WITHOUT_PROVENANCE,
        train_without,
    )

    write_jsonl(
        OUTPUT_TEST_WITHOUT_PROVENANCE,
        test_without,
    )

    write_jsonl(
        OUTPUT_TRAIN_WITH_PROVENANCE,
        train_with,
    )

    write_jsonl(
        OUTPUT_TEST_WITH_PROVENANCE,
        test_with,
    )

    print_split_statistics(
        "TRAIN — WITHOUT PROVENANCE",
        train_without,
    )

    print_split_statistics(
        "TEST — WITHOUT PROVENANCE",
        test_without,
    )

    print_split_statistics(
        "TRAIN — WITH PROVENANCE",
        train_with,
    )

    print_split_statistics(
        "TEST — WITH PROVENANCE",
        test_with,
    )

    print("\nOutput files:")
    print(f"  ✓ {OUTPUT_TRAIN_WITHOUT_PROVENANCE}")
    print(f"  ✓ {OUTPUT_TEST_WITHOUT_PROVENANCE}")
    print(f"  ✓ {OUTPUT_TRAIN_WITH_PROVENANCE}")
    print(f"  ✓ {OUTPUT_TEST_WITH_PROVENANCE}")

    print("\n" + "=" * 70)
    print("✓ STEP 54 COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(
        "\nThe dataset was split by complete trajectory with "
        "no train/test trajectory overlap."
    )


if __name__ == "__main__":
    main()