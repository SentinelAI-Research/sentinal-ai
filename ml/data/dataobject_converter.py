from backend.dataobjects.models import DataObject
from backend.dataobjects.factory import create_data_object
from ml.data.normalized_record import NormalizedDatasetRecord


class DatasetDataObjectConverter:
    """
    Converts normalized public-dataset records into SentinelAI
    DataObjects.

    Each dataset record becomes a provenance root because it is
    an original source object entering the SentinelAI pipeline.
    """

    @staticmethod
    def to_data_object(
        record: NormalizedDatasetRecord,
        created_step: str,
        object_id: str | None = None,
    ) -> DataObject:
        if not record.source_text.strip():
            raise ValueError("Cannot create a DataObject from empty source text.")

        resolved_object_id = (
            object_id
            or f"{record.source_dataset.lower().replace(' ', '-')}"
            f"-{record.record_id}"
        )

        return create_data_object(
            object_id=resolved_object_id,
            source=(f"{record.source_dataset}:" f"{record.record_id}"),
            content=record.source_text,
            sensitivity="PUBLIC",
            transformation="source_ingestion",
            parents=[],
            created_step=created_step,
        )
