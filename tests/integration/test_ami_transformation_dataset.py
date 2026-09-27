import json
from pathlib import Path

from ml.data.ami_transformation_dataset import (
    AMITransformationDatasetBuilder,
)

AMI_TRAIN_FILE = Path("data/raw/ami/train.jsonl")
OUTPUT_FILE = Path("data/processed/ami/ami_transformation_train.jsonl")


def test_build_real_ami_transformation_dataset(tmp_path: Path) -> None:
    output_path = tmp_path / "ami_transformation_train.jsonl"

    builder = AMITransformationDatasetBuilder(
        input_path=AMI_TRAIN_FILE,
        output_path=output_path,
    )

    written_records = builder.build(
        max_records=3,
    )

    assert written_records == 3
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 3

    records = [json.loads(line) for line in lines]

    assert records[0]["dataset"] == "AMI Meeting Corpus"

    assert records[0]["transformed_object"]["transformation"] == "summarize"

    assert (
        records[0]["provenance"]["parent_object_id"]
        == records[0]["source_object"]["object_id"]
    )

    assert (
        records[0]["provenance"]["child_object_id"]
        == records[0]["transformed_object"]["object_id"]
    )

    assert records[0]["provenance"]["depth"] == 1

    assert records[0]["provenance"]["transformation_depth"] == 2

    assert records[0]["transformation"]["source_text_length"] > 0

    assert records[0]["transformation"]["reference_summary_length"] > 0
