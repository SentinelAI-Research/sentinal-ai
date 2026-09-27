from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "knkarthick/AMI"
OUTPUT_DIR = Path("data/raw/ami")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset: {DATASET_NAME}")
    print(f"Target directory: {OUTPUT_DIR}")

    dataset = load_dataset(DATASET_NAME)

    print("\nDataset downloaded successfully.")
    print("\nAvailable splits:")

    for split_name, split_dataset in dataset.items():
        print(f"  - {split_name}: {len(split_dataset)} records")

    print("\nDataset features:")
    for feature_name, feature_type in dataset["train"].features.items():
        print(f"  - {feature_name}: {feature_type}")

    print("\nSaving dataset locally...")

    for split_name, split_dataset in dataset.items():
        output_file = OUTPUT_DIR / f"{split_name}.jsonl"
        split_dataset.to_json(
            str(output_file),
            orient="records",
            lines=True,
        )
        print(f"  Saved: {output_file}")

    print("\nAMI dataset download complete.")


if __name__ == "__main__":
    main()
