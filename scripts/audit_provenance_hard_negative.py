"""Read-only audit of the SentinelAI hard-negative experiment.

This script does not modify datasets, models, split files, or results.
Run from the repository root:
    python scripts/audit_provenance_hard_negative.py
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed" / "hard_negative"

LABEL_KEYS = ("label", "target", "y")
TRAJECTORY_KEYS = ("trajectory_id", "trajectory", "episode_id")
SOURCE_KEYS = ("source_dataset", "dataset", "source")
VARIANT_KEYS = ("variant", "scenario_variant", "case_type")

SUSPECT_FEATURES = (
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity",
    "max_ancestor_sensitivity_encoded",
    "sensitivity",
    "sensitivity_encoded",
    "is_external_action",
)


def first_value(record, keys):
    """Return the first available non-null value among candidate keys."""
    for key in keys:
        value = record.get(key)
        if value is not None:
            return value
    return None


def infer_variant(record):
    """Use an explicit variant field or infer it from the trajectory ID."""
    explicit = first_value(record, VARIANT_KEYS)
    if explicit is not None:
        return str(explicit)

    trajectory = first_value(record, TRAJECTORY_KEYS)
    if trajectory is None:
        return "unknown"

    trajectory = str(trajectory).lower()
    if trajectory.endswith("-provenance_risky"):
        return "provenance_risky"
    if trajectory.endswith("-benign"):
        return "benign"
    return "unknown"


def read_jsonl(path):
    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path}, line {line_number}: {exc}"
                ) from exc

            if not isinstance(item, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )
            records.append(item)
    return records


def print_counter(title, values):
    print(f"\n{title}")
    if not values:
        print("  (no values found)")
        return
    for value, count in sorted(Counter(str(v) for v in values).items()):
        print(f"  {value}: {count}")


def summarize_file(path, records):
    print("\n" + "=" * 78)
    print(f"FILE: {path.relative_to(ROOT)}")
    print(f"Rows: {len(records)}")

    if not records:
        print("  File contains no records.")
        return

    print("Fields:", ", ".join(sorted(records[0].keys())))

    labels = [first_value(r, LABEL_KEYS) for r in records]
    print_counter("Label counts", [v for v in labels if v is not None])

    variants = [infer_variant(r) for r in records]
    print_counter("Variant counts (explicit field or inferred ID)", variants)

    sources = [first_value(r, SOURCE_KEYS) for r in records]
    print_counter("Source dataset counts", [v for v in sources if v is not None])

    trajectory_ids = [
        str(value)
        for value in (first_value(r, TRAJECTORY_KEYS) for r in records)
        if value is not None
    ]
    unique_ids = set(trajectory_ids)
    print(f"\nTrajectory IDs: {len(unique_ids)} unique / {len(trajectory_ids)} rows")
    print(f"Rows missing trajectory ID: {len(records) - len(trajectory_ids)}")

    # Inspect whether candidate features have different values across labels.
    print("\nCandidate feature values by label")
    for feature in SUSPECT_FEATURES:
        present = [r for r in records if feature in r]
        if not present:
            continue

        values_by_label = defaultdict(Counter)
        for record in present:
            label = first_value(record, LABEL_KEYS)
            if label is not None:
                values_by_label[str(label)][str(record[feature])] += 1

        print(f"  {feature} ({len(present)}/{len(records)} rows contain it):")
        for label, counts in sorted(values_by_label.items()):
            compact = ", ".join(
                f"{value}={count}" for value, count in sorted(counts.items())
            )
            print(f"    {label}: {compact}")


def main():
    if not DATA_DIR.exists():
        print(f"ERROR: Dataset directory not found: {DATA_DIR}")
        raise SystemExit(1)

    files = sorted(DATA_DIR.rglob("*.jsonl"))
    if not files:
        print(f"No JSONL files found under {DATA_DIR}")
        raise SystemExit(1)

    print("SentinelAI hard-negative dataset audit")
    print(f"Repository: {ROOT}")
    print(f"Directory:  {DATA_DIR}")
    print(f"JSONL files found: {len(files)}")

    loaded = {}
    for path in files:
        try:
            loaded[path] = read_jsonl(path)
        except (OSError, ValueError) as exc:
            print(f"\nERROR reading {path.relative_to(ROOT)}: {exc}")

    for path, records in loaded.items():
        summarize_file(path, records)

    # Compare trajectory IDs between files whose names indicate train/test.
    train_files = [
        (path, rows) for path, rows in loaded.items() if "train" in path.stem.lower()
    ]
    test_files = [
        (path, rows) for path, rows in loaded.items() if "test" in path.stem.lower()
    ]

    print("\n" + "=" * 78)
    print("TRAIN / TEST TRAJECTORY OVERLAP CHECK")

    if not train_files or not test_files:
        print("Could not identify both train and test files by filename.")
        print("Review the file listing above; no split files were changed.")
        return

    for train_path, train_rows in train_files:
        train_ids = {
            str(value)
            for value in (first_value(r, TRAJECTORY_KEYS) for r in train_rows)
            if value is not None
        }

        for test_path, test_rows in test_files:
            test_ids = {
                str(value)
                for value in (first_value(r, TRAJECTORY_KEYS) for r in test_rows)
                if value is not None
            }

            overlap = train_ids & test_ids
            print(f"\nTrain: {train_path.relative_to(ROOT)}")
            print(f"Test:  {test_path.relative_to(ROOT)}")
            print(f"Unique train trajectories: {len(train_ids)}")
            print(f"Unique test trajectories:  {len(test_ids)}")
            print(f"Shared trajectory IDs:     {len(overlap)}")

            if overlap:
                print("WARNING: trajectory overlap detected. Examples:")
                for trajectory in sorted(overlap)[:10]:
                    print(f"  {trajectory}")
            else:
                print("No shared trajectory IDs detected for this pair.")

    print("\nAudit complete. No files were modified.")


if __name__ == "__main__":
    main()
