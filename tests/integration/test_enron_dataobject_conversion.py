from __future__ import annotations

from pathlib import Path
import tarfile

from backend.dataobjects.models import DataObject
from ml.data.enron_dataobject_converter import (
    EnronDataObjectConverter,
)
from ml.data.enron_normalizer import (
    EnronNormalizer,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ARCHIVE_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "enron"
    / "enron_mail_20150507.tar.gz"
)


def _get_first_email() -> tuple[str, bytes]:
    with tarfile.open(
        ARCHIVE_PATH,
        mode="r:gz",
    ) as archive:

        for member in archive.getmembers():

            if not member.isfile():
                continue

            if not member.name.startswith(
                "maildir/"
            ):
                continue

            extracted = archive.extractfile(
                member
            )

            if extracted is None:
                continue

            raw_bytes = extracted.read()

            if raw_bytes:
                return member.name, raw_bytes

    raise AssertionError(
        "No Enron email was found in the archive."
    )


def test_enron_record_converts_to_data_object() -> None:
    source_path, raw_bytes = _get_first_email()

    record = EnronNormalizer.normalize(
        raw_bytes=raw_bytes,
        source_path=source_path,
    )

    data_object = (
        EnronDataObjectConverter.to_data_object(
            record=record,
            created_step="enron-ingestion-test",
        )
    )

    assert isinstance(
        data_object,
        DataObject,
    )

    assert data_object.object_id == (
        f"enron-{record.record_id}"
    )

    assert data_object.source == (
        f"Enron Email Dataset:{record.record_id}"
    )

    assert data_object.sensitivity == "PUBLIC"

    assert data_object.transformation == (
        "source_ingestion"
    )

    assert data_object.parents == []

    assert data_object.created_step == (
        "enron-ingestion-test"
    )

    assert data_object.content_hash

    assert len(data_object.content_hash) == 64