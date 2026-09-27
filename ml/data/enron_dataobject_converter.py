from __future__ import annotations

from ml.data.enron_normalizer import NormalizedEmailRecord
from backend.dataobjects.factory import create_data_object
from backend.dataobjects.models import DataObject


class EnronDataObjectConverter:
    """
    Converts a normalized Enron email into a SentinelAI DataObject.

    The email is treated as a provenance root because it is an
    original source object rather than the output of another
    SentinelAI transformation.
    """

    DATASET_NAME = "Enron Email Dataset"

    @classmethod
    def to_data_object(
        cls,
        record: NormalizedEmailRecord,
        created_step: str,
        object_id: str | None = None,
    ) -> DataObject:

        resolved_object_id = (
            object_id
            if object_id
            else f"enron-{record.record_id}"
        )

        return create_data_object(
            object_id=resolved_object_id,
            source=f"{cls.DATASET_NAME}:{record.record_id}",
            content=record.source_text,
            sensitivity="PUBLIC",
            transformation="source_ingestion",
            parents=[],
            created_step=created_step,
        )