"""Create a variant-balanced, source-grouped hard-negative train/test split.

Safety:
- Reads the existing experiment and feature JSONL files.
- Does not overwrite existing split files.
- Keeps paired benign/risky variants of each original source trajectory
  in the same partition.
- Writes new files under data/processed/hard_negative/split_v2/.
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "hard_negative"
OUTPUT_DIR = DATA_DIR / "split_v2"

RAW_FILE = DATA_DIR / "provenance_hard_negative_experiment.jsonl"
FEATURE_FILES = {
    "without_provenance": (
        DATA_DIR / "provenance_hard_negative_features_without_provenance.jsonl"
    ),
    "with_provenance": (
        DATA_DIR / "provenance_hard_negative_features_with_provenance.jsonl"
    ),
}

TEST_FRACTION = 0.20
RANDOM_SEED = 42


def read_jsonl(path):
    if not path.exists():
        raise FileNotFoundError(f"Required input file not found: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object in {path}, line {line_number}")
            records.append(record)
    return records


def write_jsonl(path, records):
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def get_variant(record):
    variant = record.get("variant")
    if variant:
        return str(variant)

    trajectory_id = str(record.get("trajectory_id", "")).lower()
    if trajectory_id.endswith("-provenance_risky"):
        return "provenance_risky"
    if trajectory_id.endswith("-benign"):
        return "benign"
    return "unknown"


def main():
    raw_records = read_jsonl(RAW_FILE)

    # Map each generated trajectory to its original source group.
    trajectory_to_group = {}
    raw_group_variants = defaultdict(set)

    for record in raw_records:
        trajectory_id = record.get("trajectory_id")
        original_id = record.get("original_trajectory_id")
        source = record.get("source_dataset")
        variant = get_variant(record)

        if not trajectory_id or not original_id or not source:
            raise ValueError(
                "Raw records must contain trajectory_id, "
                "original_trajectory_id, and source_dataset."
            )

        group_key = (str(source), str(original_id))
        trajectory_id = str(trajectory_id)

        existing = trajectory_to_group.get(trajectory_id)
        if existing is not None and existing != group_key:
            raise ValueError(
                f"Trajectory {trajectory_id} maps to multiple source groups."
            )

        trajectory_to_group[trajectory_id] = group_key
        raw_group_variants[group_key].add(variant)

    if not raw_group_variants:
        raise ValueError("No original source groups found.")

    # Each original source should have both paired variants.
    incomplete_groups = {
        group: variants
        for group, variants in raw_group_variants.items()
        if variants != {"benign", "provenance_risky"}
    }
    if incomplete_groups:
        examples = list(incomplete_groups.items())[:5]
        raise ValueError(
            "Some original source groups do not contain exactly the expected "
            f"paired variants. Examples: {examples}"
        )

    # Split original groups separately within each source dataset.
    # This maintains the source mix and keeps both paired variants together.
    groups_by_source = defaultdict(list)
    for group_key in raw_group_variants:
        source, _original_id = group_key
        groups_by_source[source].append(group_key)

    rng = random.Random(RANDOM_SEED)
    test_groups = set()

    for source in sorted(groups_by_source):
        groups = sorted(groups_by_source[source])
        rng.shuffle(groups)

        if len(groups) < 2:
            raise ValueError(f"Not enough original groups for source {source!r}.")

        number_test = max(1, round(len(groups) * TEST_FRACTION))
        test_groups.update(groups[:number_test])

        print(
            f"Source {source}: {len(groups)} original groups; "
            f"{len(groups) - number_test} train groups; "
            f"{number_test} test groups"
        )

    train_groups = set(raw_group_variants) - test_groups

    if train_groups & test_groups:
        raise AssertionError("Original source groups overlap across partitions.")

    # Build the trajectory-level mapping used to partition feature records.
    train_trajectory_ids = {
        trajectory_id
        for trajectory_id, group in trajectory_to_group.items()
        if group in train_groups
    }
    test_trajectory_ids = {
        trajectory_id
        for trajectory_id, group in trajectory_to_group.items()
        if group in test_groups
    }

    if train_trajectory_ids & test_trajectory_ids:
        raise AssertionError("Trajectory IDs overlap across partitions.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    expected_trajectory_ids = set(trajectory_to_group)
    summary = {
        "random_seed": RANDOM_SEED,
        "test_fraction": TEST_FRACTION,
        "train_original_groups": len(train_groups),
        "test_original_groups": len(test_groups),
        "train_trajectories": len(train_trajectory_ids),
        "test_trajectories": len(test_trajectory_ids),
        "files": {},
    }

    for feature_name, feature_path in FEATURE_FILES.items():
        records = read_jsonl(feature_path)

        train_records = []
        test_records = []
        unknown_ids = set()

        for record in records:
            trajectory_id = record.get("trajectory_id")
            if trajectory_id is None:
                raise ValueError(f"Missing trajectory_id in {feature_path.name}")

            trajectory_id = str(trajectory_id)

            if trajectory_id in train_trajectory_ids:
                train_records.append(record)
            elif trajectory_id in test_trajectory_ids:
                test_records.append(record)
            else:
                unknown_ids.add(trajectory_id)

        if unknown_ids:
            raise ValueError(
                f"{feature_path.name} contains trajectory IDs missing from "
                f"the raw experiment. Examples: {sorted(unknown_ids)[:5]}"
            )

        observed_ids = {str(record["trajectory_id"]) for record in records}
        if observed_ids != expected_trajectory_ids:
            raise ValueError(
                f"Trajectory ID mismatch in {feature_path.name}: "
                f"{len(observed_ids)} IDs in features versus "
                f"{len(expected_trajectory_ids)} in raw experiment."
            )

        suffix = feature_name
        train_path = OUTPUT_DIR / f"train_{suffix}.jsonl"
        test_path = OUTPUT_DIR / f"test_{suffix}.jsonl"

        # New output paths only; fail rather than overwrite an earlier run.
        if train_path.exists() or test_path.exists():
            raise FileExistsError(
                f"Output files already exist in {OUTPUT_DIR}. "
                "Inspect them first; this script will not overwrite them."
            )

        write_jsonl(train_path, train_records)
        write_jsonl(test_path, test_records)

        # Validate variant balance in each partition.
        partition_counts = {}
        for name, partition in (
            ("train", train_records),
            ("test", test_records),
        ):
            variants = Counter(get_variant(record) for record in partition)
            partition_counts[name] = dict(variants)

            if variants.get("benign", 0) != variants.get("provenance_risky", 0):
                raise AssertionError(
                    f"{feature_name}: {name} rows are not balanced by variant: "
                    f"{dict(variants)}"
                )

            partition_groups = set()
            for record in partition:
                tid = str(record["trajectory_id"])
                partition_groups.add(trajectory_to_group[tid])

            expected_groups = train_groups if name == "train" else test_groups
            if partition_groups != expected_groups:
                raise AssertionError(
                    f"{feature_name}: unexpected source groups in {name} set."
                )

        summary["files"][feature_name] = {
            "train_rows": len(train_records),
            "test_rows": len(test_records),
            "train_variant_counts": partition_counts["train"],
            "test_variant_counts": partition_counts["test"],
            "train_labels": dict(
                Counter(str(record.get("label", "MISSING")) for record in train_records)
            ),
            "test_labels": dict(
                Counter(str(record.get("label", "MISSING")) for record in test_records)
            ),
            "train_file": str(train_path.relative_to(ROOT)),
            "test_file": str(test_path.relative_to(ROOT)),
        }

    summary_path = OUTPUT_DIR / "split_summary.json"
    if summary_path.exists():
        raise FileExistsError(
            f"{summary_path} already exists; refusing to overwrite it."
        )

    write_jsonl(
        OUTPUT_DIR / "split_validation.jsonl",
        [
            {
                "status": "PASS",
                "train_trajectory_overlap": 0,
                "train_test_original_group_overlap": 0,
                "variant_balance_checked": True,
                "source_grouping_checked": True,
            }
        ],
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("\nNew split created and basic checks passed.")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Original groups: {len(raw_group_variants)}")
    print(f"Train original groups: {len(train_groups)}")
    print(f"Test original groups: {len(test_groups)}")
    print(f"Train trajectories: {len(train_trajectory_ids)}")
    print(f"Test trajectories: {len(test_trajectory_ids)}")
    print("\nVariant counts, labels, and output paths:")
    print(json.dumps(summary["files"], indent=2))
    print("\nExisting split files and trained models were not modified.")


if __name__ == "__main__":
    main()
