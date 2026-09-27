from __future__ import annotations

import tempfile
from pathlib import Path

from backend.dataobjects.factory import create_data_object
from backend.provenance.graph import ProvenanceGraph
from backend.tools.file_tools import paraphrase, summarize
from backend.tools.memory_tools import read_memory, write_memory
from ml.data.enron_normalizer import EnronNormalizer, NormalizedEmailRecord


class EnronMemoryTrajectoryService:
    """
    Builds a provenance-aware memory-mediated trajectory from
    a real Enron email.

    The trajectory can be built either from an archive member or
    directly from an already-normalized email record.

    Each trajectory receives its own isolated temporary memory file.
    """

    @classmethod
    def build_from_archive_member(
        cls,
        archive_path: str | Path,
        member_path: str,
    ) -> tuple[ProvenanceGraph, list[str]]:
        """
        Backward-compatible helper.

        Reads one email from the compressed archive and delegates
        trajectory construction to build_from_record().
        """

        import tarfile

        archive = Path(archive_path)

        if not archive.exists():
            raise FileNotFoundError(
                f"Enron archive not found: {archive}"
            )

        with tarfile.open(archive, mode="r:gz") as tar:
            try:
                member = tar.getmember(member_path)
            except KeyError as exc:
                raise FileNotFoundError(
                    f"Archive member not found: {member_path}"
                ) from exc

            if not member.isfile():
                raise ValueError(
                    f"Archive member is not a file: {member_path}"
                )

            extracted = tar.extractfile(member)

            if extracted is None:
                raise RuntimeError(
                    f"Could not read archive member: {member_path}"
                )

            raw_bytes = extracted.read()

        return cls.build_from_raw_email(
            raw_bytes=raw_bytes,
            source_path=member_path,
        )

    @classmethod
    def build_from_raw_email(
        cls,
        raw_bytes: bytes,
        source_path: str,
    ) -> tuple[ProvenanceGraph, list[str]]:
        """
        Normalize one raw Enron email and build its trajectory.

        This method is designed for the experiment builder so the
        compressed archive can be opened only once.
        """

        if not isinstance(raw_bytes, bytes):
            raise TypeError(
                "raw_bytes must be bytes, "
                f"got {type(raw_bytes).__name__}"
            )

        if not raw_bytes.strip():
            raise ValueError(
                f"Email content is empty: {source_path}"
            )

        record = EnronNormalizer.normalize(
            raw_bytes=raw_bytes,
            source_path=source_path,
        )

        return cls.build_from_record(record)

    @classmethod
    def build_from_record(
        cls,
        record: NormalizedEmailRecord,
    ) -> tuple[ProvenanceGraph, list[str]]:
        """
        Build a complete provenance-aware trajectory from an
        already-normalized Enron email.

        No archive access occurs here.
        """

        if not isinstance(record, NormalizedEmailRecord):
            raise TypeError(
                "record must be a NormalizedEmailRecord, "
                f"got {type(record).__name__}"
            )

        source_id = f"enron-{record.record_id}"

        graph = ProvenanceGraph()

        # ---------------------------------------------------------
        # ISOLATED MEMORY
        # ---------------------------------------------------------

        with tempfile.TemporaryDirectory(
            prefix="sentinelai-enron-memory-"
        ) as temp_dir:

            memory_path = Path(temp_dir) / "memory.txt"

            # -----------------------------------------------------
            # SOURCE EMAIL
            # -----------------------------------------------------

            source_object = create_data_object(
                object_id=source_id,
                source=f"Enron Email Dataset:{record.record_id}",
                content=record.source_text,
                sensitivity="PUBLIC",
                transformation="source_ingestion",
                parents=[],
                created_step="enron-source-ingestion",
            )

            graph.add_object(source_object)

            # -----------------------------------------------------
            # STEP 1 — SUMMARIZATION
            # -----------------------------------------------------

            summary_result = summarize(
                text=record.source_text
            )

            if not isinstance(summary_result, str):
                raise TypeError(
                    "summarize() must return str, "
                    f"got {type(summary_result).__name__}"
                )

            if not summary_result.strip():
                raise ValueError(
                    "summarize() returned empty content."
                )

            summary_id = f"{source_id}-summary"

            summary_object = create_data_object(
                object_id=summary_id,
                source=source_object.source,
                content=summary_result,
                sensitivity=source_object.sensitivity,
                transformation="summarize",
                parents=[source_id],
                created_step="enron-summary",
            )

            graph.add_object(summary_object)
            graph.add_relationship(
                source_id,
                summary_id,
            )

            # -----------------------------------------------------
            # STEP 2 — MEMORY WRITE
            # -----------------------------------------------------

            memory_content = write_memory(
                content=summary_result,
                memory_path=memory_path,
            )

            if not isinstance(memory_content, str):
                raise TypeError(
                    "write_memory() must return str, "
                    f"got {type(memory_content).__name__}"
                )

            if not memory_content.strip():
                raise ValueError(
                    "write_memory() returned empty content."
                )

            memory_id = f"{source_id}-memory"

            memory_object = create_data_object(
                object_id=memory_id,
                source=source_object.source,
                content=memory_content,
                sensitivity=source_object.sensitivity,
                transformation="write_memory",
                parents=[summary_id],
                created_step="enron-memory-write",
            )

            graph.add_object(memory_object)
            graph.add_relationship(
                summary_id,
                memory_id,
            )

            # -----------------------------------------------------
            # STEP 3 — MEMORY READ
            # -----------------------------------------------------

            retrieved_content = read_memory(
                memory_path=memory_path,
            )

            if not isinstance(retrieved_content, str):
                raise TypeError(
                    "read_memory() must return str, "
                    f"got {type(retrieved_content).__name__}"
                )

            if not retrieved_content.strip():
                raise ValueError(
                    "read_memory() returned empty content."
                )

            retrieved_id = f"{source_id}-retrieved"

            retrieved_object = create_data_object(
                object_id=retrieved_id,
                source=source_object.source,
                content=retrieved_content,
                sensitivity=source_object.sensitivity,
                transformation="read_memory",
                parents=[memory_id],
                created_step="enron-memory-read",
            )

            graph.add_object(retrieved_object)
            graph.add_relationship(
                memory_id,
                retrieved_id,
            )

            # -----------------------------------------------------
            # STEP 4 — PARAPHRASING
            # -----------------------------------------------------

            paraphrase_result = paraphrase(
                text=retrieved_content
            )

            if not isinstance(paraphrase_result, str):
                raise TypeError(
                    "paraphrase() must return str, "
                    f"got {type(paraphrase_result).__name__}"
                )

            if not paraphrase_result.strip():
                raise ValueError(
                    "paraphrase() returned empty content."
                )

            paraphrase_id = f"{source_id}-paraphrase"

            paraphrase_object = create_data_object(
                object_id=paraphrase_id,
                source=source_object.source,
                content=paraphrase_result,
                sensitivity=source_object.sensitivity,
                transformation="paraphrase",
                parents=[retrieved_id],
                created_step="enron-paraphrase",
            )

            graph.add_object(paraphrase_object)
            graph.add_relationship(
                retrieved_id,
                paraphrase_id,
            )

            # -----------------------------------------------------
            # STEP 5 — MESSAGE PREPARATION
            # -----------------------------------------------------

            message_id = f"{source_id}-message"

            message_object = create_data_object(
                object_id=message_id,
                source=source_object.source,
                content=paraphrase_result,
                sensitivity=source_object.sensitivity,
                transformation="send_message",
                parents=[paraphrase_id],
                created_step="enron-send-message",
            )

            graph.add_object(message_object)
            graph.add_relationship(
                paraphrase_id,
                message_id,
            )

            trajectory = [
                source_id,
                summary_id,
                memory_id,
                retrieved_id,
                paraphrase_id,
                message_id,
            ]

            return graph, trajectory