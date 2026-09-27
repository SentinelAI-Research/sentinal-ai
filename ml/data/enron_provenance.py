from __future__ import annotations

import tarfile
from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from ml.data.enron_dataobject_converter import EnronDataObjectConverter
from ml.data.enron_normalizer import EnronNormalizer


class EnronProvenanceLoader:
    """
    Loads real Enron emails directly from the compressed archive and
    registers them as root DataObjects in a provenance graph.

    The archive is intentionally not fully extracted because the complete
    Enron dataset is very large.
    """

    ARCHIVE_NAME = "enron_mail_20150507.tar.gz"

    @classmethod
    def load_archive(
        cls,
        archive_path: str | Path,
        max_records: int = 100,
    ) -> tuple[ProvenanceGraph, list]:
        """
        Load up to max_records email files from the Enron archive.

        The archive is processed incrementally. No full extraction is
        performed.
        """

        if max_records <= 0:
            raise ValueError("max_records must be greater than zero.")

        archive = Path(archive_path)

        if not archive.exists():
            raise FileNotFoundError(
                f"Enron archive not found: {archive}"
            )

        graph = ProvenanceGraph()
        normalized_records = []

        candidate_count = 0

        with tarfile.open(archive, mode="r:gz") as tar:
            for member in tar:
                if len(normalized_records) >= max_records:
                    break

                if not member.isfile():
                    continue

                if not cls._looks_like_email(member.name):
                    continue

                candidate_count += 1

                extracted = tar.extractfile(member)

                if extracted is None:
                    raise RuntimeError(
                        f"Could not read Enron archive member: "
                        f"{member.name}"
                    )

                raw_bytes = extracted.read()

                if not raw_bytes.strip():
                    continue

                try:
                    record = EnronNormalizer.normalize(
                        raw_bytes=raw_bytes,
                        source_path=member.name,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        "Failed to normalize an Enron email.\n"
                        f"Archive member: {member.name}\n"
                        f"Candidate number: {candidate_count}\n"
                        f"Original error: {type(exc).__name__}: {exc}"
                    ) from exc

                object_id = f"enron-{record.record_id}"

                data_object = EnronDataObjectConverter.to_data_object(
                    record=record,
                    created_step="enron-source-ingestion",
                    object_id=object_id,
                )

                graph.add_object(data_object)
                normalized_records.append(record)

        if not normalized_records:
            raise RuntimeError(
                "No Enron email records were loaded.\n"
                f"Archive: {archive}\n"
                f"Email candidates detected: {candidate_count}\n"
                "Check the archive structure and EnronNormalizer."
            )

        return graph, normalized_records

    @staticmethod
    def _looks_like_email(path: str) -> bool:
        """
        Determine whether an archive member represents an Enron email.

        Enron messages are stored beneath:

            maildir/<user>/<folder>/<message>

        We intentionally do not use the filename extension to identify
        messages because Enron folder/message names can contain dots.
        """

        normalized = path.replace("\\", "/").strip("/")

        parts = normalized.split("/")

        # Expected minimum structure:
        #
        # maildir
        #   user
        #     folder
        #       message
        #
        if len(parts) < 4:
            return False

        if parts[0].lower() != "maildir":
            return False

        # A valid message must have a non-empty message component.
        if not parts[-1]:
            return False

        # Reject obvious directory-like members.
        if path.endswith("/"):
            return False

        return True