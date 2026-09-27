from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from ml.features.ami_transformation_features import (
    AMITransformationFeatureBuilder,
)

INPUT_FILE = (
    PROJECT_ROOT / "data" / "processed" / "ami" / "ami_transformation_train.jsonl"
)

OUTPUT_FILE = PROJECT_ROOT / "data" / "processed" / "ami" / "ami_ml_features.jsonl"


def main() -> None:
    count = AMITransformationFeatureBuilder.build_file(
        input_path=INPUT_FILE,
        output_path=OUTPUT_FILE,
    )

    print(f"Successfully generated ML features " f"for {count} records.")

    print(f"Output: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
