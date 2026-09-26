import hashlib
from datetime import datetime, timezone

from backend.dataobjects.models import DataObject

"""
Standard way to create DataObjects

For eg.
real file content
      ↓
SHA-256 hash
      ↓
DataObject

The hash lets us identify the exact content represented by the object without storing the raw content inside the provenance graph.
"""
def create_data_object(
    object_id: str,
    source: str,
    content: str,
    sensitivity: str,
    transformation: str | None,
    parents: list[str],
    created_step: str,
) -> DataObject:
    content_hash = hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

    return DataObject(
        object_id=object_id,
        source=source,
        sensitivity=sensitivity,
        parents=parents,
        transformation=transformation,
        content_hash=content_hash,
        embedding_ref=None,
        created_at=datetime.now(timezone.utc),
        created_step=created_step,
    )