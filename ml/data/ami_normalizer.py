import json
from pathlib import Path

from ml.data.normalized_record import NormalizedDatasetRecord


class AMINormalizer:
    """
    Converts raw AMI JSONL records into the common
    NormalizedDatasetRecord representation.
    """

    DATASET_NAME = "AMI Meeting Corpus"

    @classmethod
    def normalize_record(
        cls,
        record: dict,
    ) -> NormalizedDatasetRecord:
        required_fields = ["id", "dialogue", "summary"]

        missing_fields = [field for field in required_fields if field not in record]

        if missing_fields:
            raise ValueError(
                f"AMI record is missing required fields: " f"{missing_fields}"
            )

        record_id = str(record["id"])
        dialogue = str(record["dialogue"]).strip()
        summary = str(record["summary"]).strip()

        if not dialogue:
            raise ValueError(f"AMI record {record_id} has empty dialogue.")

        if not summary:
            raise ValueError(f"AMI record {record_id} has empty summary.")

        return NormalizedDatasetRecord(
            record_id=record_id,
            source_dataset=cls.DATASET_NAME,
            source_text=dialogue,
            reference_text=summary,
            metadata={
                "dataset_record_id": record_id,
                "source_format": "jsonl",
            },
        )

    @classmethod
    def normalize_file(
        cls,
        input_path: str | Path,
    ) -> list[NormalizedDatasetRecord]:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"AMI file does not exist: {input_path}")

        records = []

        with input_path.open(
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
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line " f"{line_number} in {input_path}: {exc}"
                    ) from exc

                normalized = cls.normalize_record(raw_record)

                records.append(normalized)

        return records
