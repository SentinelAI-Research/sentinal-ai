import json
from pathlib import Path

from ml.features.ami_transformation_features import (
    AMITransformationFeatureBuilder,
)

INPUT_FILE = Path("data/processed/ami/ami_transformation_train.jsonl")


def test_ami_transformation_features() -> None:
    with INPUT_FILE.open(
        "r",
        encoding="utf-8",
    ) as file:
        first_record = json.loads(file.readline())

    features = AMITransformationFeatureBuilder.build_record_features(first_record)

    assert features["record_id"]

    assert features["source_dataset"] == "AMI Meeting Corpus"

    assert features["source_text_length"] > 0

    assert features["reference_summary_length"] > 0

    assert features["compression_ratio"] > 0

    assert features["provenance_depth"] == 1

    assert features["transformation_depth"] == 2

    assert features["has_parent"] == 1

    assert features["source_sensitivity"] == "PUBLIC"

    assert features["transformation_name"] == "summarize"

    assert features["source_content_hash"]

    assert features["output_content_hash"]
