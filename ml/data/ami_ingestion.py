import json
from pathlib import Path

from backend.dataobjects.models import DataObject
from ml.data.ami_normalizer import AMINormalizer
from ml.data.dataobject_converter import (
    DatasetDataObjectConverter,
)


class AMIIngestionPipeline:
    """
    Ingests real AMI JSONL records into SentinelAI
    DataObjects.

    The original AMI dialogue becomes the content of a
    provenance-root DataObject.
    """

    def __init__(self) -> None:
        self.normalizer = AMINormalizer()

    def ingest_file(
        self,
        input_path: str | Path,
        split_name: str,
        max_records: int | None = None,
    ) -> list[DataObject]:
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"AMI dataset file does not exist: {input_path}")

        if max_records is not None and max_records <= 0:
            raise ValueError("max_records must be greater than zero.")

        data_objects: list[DataObject] = []

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            for line_number, line in enumerate(
                file,
                start=1,
            ):
                if max_records is not None and len(data_objects) >= max_records:
                    break

                line = line.strip()

                if not line:
                    continue

                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line " f"{line_number} in {input_path}: {exc}"
                    ) from exc

                normalized = self.normalizer.normalize_record(raw_record)

                object_id = f"ami-{split_name}-" f"{normalized.record_id}"

                data_object = DatasetDataObjectConverter.to_data_object(
                    record=normalized,
                    created_step=(f"ami-ingestion-{split_name}"),
                    object_id=object_id,
                )

                data_objects.append(data_object)

        return data_objects
