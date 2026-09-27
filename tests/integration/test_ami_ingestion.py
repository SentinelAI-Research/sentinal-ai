from pathlib import Path

from backend.dataobjects.models import DataObject
from ml.data.ami_ingestion import AMIIngestionPipeline
from ml.data.ami_normalizer import AMINormalizer

AMI_TRAIN_FILE = Path("data/raw/ami/train.jsonl")


def test_ami_ingestion_creates_real_data_objects() -> None:
    pipeline = AMIIngestionPipeline()

    objects = pipeline.ingest_file(
        input_path=AMI_TRAIN_FILE,
        split_name="train",
        max_records=5,
    )

    expected_records = AMINormalizer.normalize_file(AMI_TRAIN_FILE)[:5]

    assert len(objects) == 5

    for data_object, record in zip(
        objects,
        expected_records,
    ):
        assert isinstance(
            data_object,
            DataObject,
        )

        assert data_object.object_id == (f"ami-train-{record.record_id}")

        assert data_object.source == (f"AMI Meeting Corpus:{record.record_id}")

        assert data_object.sensitivity == "PUBLIC"

        assert data_object.transformation == ("source_ingestion")

        assert data_object.parents == []

        assert data_object.content_hash

        assert data_object.created_step == ("ami-ingestion-train")
