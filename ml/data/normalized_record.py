from pydantic import BaseModel, Field


class NormalizedDatasetRecord(BaseModel):
    """
    Common representation of a record from a public dataset.

    Dataset-specific loaders convert their original schema into
    this representation before the record enters the SentinelAI
    processing pipeline.
    """

    record_id: str = Field(min_length=1)

    source_dataset: str = Field(min_length=1)

    source_text: str = Field(min_length=1)

    reference_text: str | None = None

    metadata: dict[str, str] = Field(default_factory=dict)
