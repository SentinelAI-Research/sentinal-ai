import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ami" / "ami_transformation_train.jsonl"
)


def main() -> None:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Dataset does not exist: {INPUT_FILE}")

    records = []

    with INPUT_FILE.open(
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
                    f"Invalid JSON at line " f"{line_number}: {exc}"
                ) from exc

            records.append(record)

    record_ids = set()
    source_object_ids = set()
    transformed_object_ids = set()

    invalid_records = []

    for record in records:
        record_id = record["record_id"]

        source_object_id = record["source_object"]["object_id"]

        transformed_object_id = record["transformed_object"]["object_id"]

        parent_id = record["provenance"]["parent_object_id"]

        child_id = record["provenance"]["child_object_id"]

        if record_id in record_ids:
            invalid_records.append(f"Duplicate record ID: {record_id}")

        if source_object_id in source_object_ids:
            invalid_records.append(f"Duplicate source object: " f"{source_object_id}")

        if transformed_object_id in transformed_object_ids:
            invalid_records.append(
                f"Duplicate transformed object: " f"{transformed_object_id}"
            )

        if parent_id != source_object_id:
            invalid_records.append(f"Broken parent link for {record_id}")

        if child_id != transformed_object_id:
            invalid_records.append(f"Broken child link for {record_id}")

        if record["provenance"]["depth"] != 1:
            invalid_records.append(f"Unexpected provenance depth " f"for {record_id}")

        if record["transformation"]["source_text_length"] <= 0:
            invalid_records.append(f"Empty source text for {record_id}")

        if record["transformation"]["reference_summary_length"] <= 0:
            invalid_records.append(f"Empty reference summary " f"for {record_id}")

        record_ids.add(record_id)
        source_object_ids.add(source_object_id)
        transformed_object_ids.add(transformed_object_id)

    print("=" * 60)
    print("AMI TRANSFORMATION DATASET INSPECTION")
    print("=" * 60)

    print(f"Dataset: {INPUT_FILE}")
    print(f"Records: {len(records)}")
    print(f"Unique record IDs: " f"{len(record_ids)}")
    print(f"Unique source objects: " f"{len(source_object_ids)}")
    print(f"Unique transformed objects: " f"{len(transformed_object_ids)}")

    if invalid_records:
        print()
        print("INVALID RECORDS:")

        for error in invalid_records:
            print(f"- {error}")

        raise SystemExit(
            f"\nDataset validation failed with " f"{len(invalid_records)} issue(s)."
        )

    print()
    print("VALIDATION: PASSED")
    print("All provenance relationships " "are consistent.")
    print("No duplicate record/object IDs detected.")
    print("All source and reference texts " "are non-empty.")
    print("=" * 60)


if __name__ == "__main__":
    main()
