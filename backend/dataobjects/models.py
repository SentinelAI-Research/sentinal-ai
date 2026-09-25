from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class DataObject(BaseModel):
    """
    Represents a piece of data flowing through SentinelAI.

    A DataObject can be:
    - a source object (e.g. a file)
    - a transformed object (e.g. a summary)
    - a memory object
    - a calculated/derived object
    - an output sent to another destination
    """

    object_id: str
    source: str
    sensitivity: str

    parents: list[str] = Field(default_factory=list)

    transformation: Optional[str] = None

    content_hash: Optional[str] = None

    embedding_ref: Optional[str] = None

    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    created_step: Optional[str] = None