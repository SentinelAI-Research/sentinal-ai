import json
from pathlib import Path

AMI_DIR = Path("data/raw/ami")


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

    return records


def inspect_split(split_name: str) -> None:
    path = AMI_DIR / f"{split_name}.jsonl"

    if not path.exists():
        raise FileNotFoundError(f"Dataset split not found: {path}")

    records = load_jsonl(path)

    print(f"\n{'=' * 60}")
    print(f"Split: {split_name}")
    print(f"{'=' * 60}")

    print(f"Records: {len(records)}")

    if not records:
        print("No records found.")
        return

    fields = sorted({field for record in records for field in record.keys()})

    print(f"Fields: {fields}")

    print("\nMissing values:")

    for field in fields:
        missing = sum(
            1
            for record in records
            if record.get(field) is None or record.get(field) == ""
        )

        print(f"  {field}: {missing}")

    dialogue_lengths = [len(record.get("dialogue", "")) for record in records]

    summary_lengths = [len(record.get("summary", "")) for record in records]

    print("\nText statistics:")

    print(
        f"  Dialogue characters: "
        f"min={min(dialogue_lengths)}, "
        f"max={max(dialogue_lengths)}, "
        f"average={sum(dialogue_lengths) / len(dialogue_lengths):.2f}"
    )

    print(
        f"  Summary characters: "
        f"min={min(summary_lengths)}, "
        f"max={max(summary_lengths)}, "
        f"average={sum(summary_lengths) / len(summary_lengths):.2f}"
    )

    print("\nFirst record:")

    first_record = records[0]

    print(f"  ID: {first_record.get('id')}")
    print(f"  Dialogue preview: " f"{first_record.get('dialogue', '')[:300]}...")
    print(f"  Summary preview: " f"{first_record.get('summary', '')[:300]}...")


def main() -> None:
    print("SentinelAI - AMI Dataset Inspection")

    for split_name in ["train", "validation", "test"]:
        inspect_split(split_name)

    print("\nAMI dataset inspection completed successfully.")


if __name__ == "__main__":
    main()
