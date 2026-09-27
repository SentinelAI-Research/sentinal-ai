import json
from pathlib import Path

from ml.features.enron_security_features import (
    EnronSecurityFeatureBuilder,
)


ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_train.jsonl"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_features.jsonl"
)


def test_enron_security_feature_builder():

    generated = (
        EnronSecurityFeatureBuilder.build_file(
            input_path=INPUT_PATH,
            output_path=OUTPUT_PATH,
        )
    )

    assert generated == 12

    assert OUTPUT_PATH.exists()

    with OUTPUT_PATH.open(
        "r",
        encoding="utf-8",
    ) as file:

        records = [
            json.loads(line)
            for line in file
            if line.strip()
        ]

    assert len(records) == 12

    first = records[0]
    last = records[5]

    assert first["step_number"] == 1
    assert first["tool_name"] == "read_file"
    assert first["tool_encoded"] == 0
    assert first["provenance_depth"] == 0
    assert first["transformation_depth"] == 1
    assert first["trajectory_progress"] == 1 / 6
    assert first["label"] == "SAFE"
    assert first["label_encoded"] == 0

    assert last["step_number"] == 6
    assert last["tool_name"] == "send_message"
    assert last["tool_encoded"] == 5
    assert last["provenance_depth"] == 5
    assert last["transformation_depth"] == 6
    assert last["trajectory_progress"] == 1.0
    assert last["label"] == "VIOLATION"
    assert last["label_encoded"] == 2

    memory_records = [
        record
        for record in records
        if record["is_memory_operation"] == 1
    ]

    assert len(memory_records) == 4