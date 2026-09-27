from pathlib import Path

from backend.dataobjects.models import DataObject
from ml.data.ami_normalizer import AMINormalizer
from ml.data.dataobject_converter import (
    DatasetDataObjectConverter,
)


AMI_TRAIN_FILE = Path(
    "data/raw/ami/train.jsonl"
)


def test_ami_record_becomes_data_object() -> None:
    records = AMINormalizer.normalize_file(
        AMI_TRAIN_FILE
    )

    assert records

    record = records[0]

    data_object = (
        DatasetDataObjectConverter.to_data_object(
            record=record,
            created_step="ami-ingestion-step-1",
        )
    )

    assert isinstance(
        data_object,
        DataObject,
    )

    assert data_object.object_id
    assert data_object.source.startswith(
        "AMI Meeting Corpus:"
    )
    assert data_object.sensitivity == "PUBLIC"
    assert data_object.transformation == (
        "source_ingestion"
    )
    assert data_object.parents == []
    assert data_object.content_hash
    assert data_object.created_step == (
        "ami-ingestion-step-1"
    )