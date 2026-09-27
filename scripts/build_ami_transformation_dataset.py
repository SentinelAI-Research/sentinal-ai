from pathlib import Path
import sys

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.data.ami_transformation_dataset import (
    AMITransformationDatasetBuilder,
)

INPUT_FILE = PROJECT_ROOT / "data" / "raw" / "ami" / "train.jsonl"

OUTPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ami" / "ami_transformation_train.jsonl"
)

MAX_RECORDS = 100


def main() -> None:
    builder = AMITransformationDatasetBuilder(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
    )

    written_records = builder.build(
        max_records=MAX_RECORDS,
    )

    print(f"Successfully generated {written_records} " f"AMI transformation records.")

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
