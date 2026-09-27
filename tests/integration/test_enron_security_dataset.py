import json
from pathlib import Path

from ml.data.enron_security_dataset import (
    EnronSecurityDatasetBuilder,
)


ARCHIVE_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "enron"
    / "enron_mail_20150507.tar.gz"
)

OUTPUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "processed"
    / "enron"
    / "enron_security_train.jsonl"
)


def test_enron_security_dataset_builder():

    generated = (
        EnronSecurityDatasetBuilder.build_from_archive(
            archive_path=ARCHIVE_PATH,
            output_path=OUTPUT_PATH,
            max_records=2,
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

    first_trajectory = records[:6]

    assert [
        record["step_number"]
        for record in first_trajectory
    ] == [1, 2, 3, 4, 5, 6]

    assert [
        record["tool_name"]
        for record in first_trajectory
    ] == [
        "read_file",
        "summarize",
        "write_memory",
        "read_memory",
        "paraphrase",
        "send_message",
    ]

    assert [
        record["label"]
        for record in first_trajectory
    ] == [
        "SAFE",
        "EARLY_RISK",
        "EARLY_RISK",
        "EARLY_RISK",
        "EARLY_RISK",
        "VIOLATION",
    ]

    assert first_trajectory[0][
        "provenance_depth"
    ] == 0

    assert first_trajectory[-1][
        "provenance_depth"
    ] == 5

    assert first_trajectory[0][
        "transformation_depth"
    ] == 1

    assert first_trajectory[-1][
        "transformation_depth"
    ] == 6