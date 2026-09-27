from __future__ import annotations

from pathlib import Path
import tarfile

from ml.data.enron_normalizer import (
    EnronNormalizer,
    NormalizedEmailRecord,
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


def test_enron_normalizer_parses_real_email() -> None:
    source_path, raw_bytes = _get_first_email()

    record = EnronNormalizer.normalize(
        raw_bytes=raw_bytes,
        source_path=source_path,
    )

    assert isinstance(
        record,
        NormalizedEmailRecord,
    )

    assert record.source_dataset == "enron"

    assert record.record_id == source_path

    assert record.source_path == source_path

    assert record.user

    assert record.folder

    assert record.source_text.strip()

    assert record.subject is not None

    assert (
        record.message_id is not None
        or record.sender is not None
    )


def test_enron_normalizer_preserves_metadata() -> None:
    source_path, raw_bytes = _get_first_email()

    record = EnronNormalizer.normalize(
        raw_bytes=raw_bytes,
        source_path=source_path,
    )

    assert "x_from" in record.metadata
    assert "x_to" in record.metadata
    assert "x_folder" in record.metadata
    assert "x_origin" in record.metadata
    assert "x_filename" in record.metadata