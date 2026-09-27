from __future__ import annotations

import json
import tarfile
from pathlib import Path

from ml.data.enron_memory_trajectory import (
    EnronMemoryTrajectoryService,
)
from ml.data.enron_normalizer import EnronNormalizer


class EnronSecurityDatasetBuilder:
    """
    Builds security-labeled trajectory records from real Enron emails.

    The compressed Enron archive is opened exactly once.

    Each valid email is normalized and transformed immediately,
    avoiding repeated scans of the large compressed archive.
    """

    SCENARIO_ID = "enron-memory-mediated-leakage"
    SCENARIO_TYPE = "LONG_HORIZON_LEAKAGE"

    STEP_LABELS = {
        1: "SAFE",
        2: "EARLY_RISK",
        3: "EARLY_RISK",
        4: "EARLY_RISK",
        5: "EARLY_RISK",
        6: "VIOLATION",
    }

    TOOL_NAMES = {
        1: "read_file",
        2: "summarize",
        3: "write_memory",
        4: "read_memory",
        5: "paraphrase",
        6: "send_message",
    }

    @classmethod
    def build_from_archive(
        cls,
        archive_path: str | Path,
        output_path: str | Path,
        max_records: int = 100,
    ) -> int:
        """
        Build labeled trajectory records from real Enron emails.

        The archive is opened once and processed sequentially.

        Args:
            archive_path: Path to the compressed Enron archive.
            output_path: JSONL output path.
            max_records: Maximum number of email trajectories.

        Returns:
            Number of generated trajectory-step records.
        """

        archive = Path(archive_path)
        output = Path(output_path)

        if not archive.exists():
            raise FileNotFoundError(
                f"Enron archive not found: {archive}"
            )

        if max_records <= 0:
            raise ValueError(
                "max_records must be greater than zero."
            )

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        total_records = 0
        trajectories_processed = 0

        print(
            f"Opening Enron archive once: {archive}"
        )

        with tarfile.open(
            archive,
            mode="r:gz",
        ) as tar, output.open(
            "w",
            encoding="utf-8",
        ) as file:

            for member in tar:

                if trajectories_processed >= max_records:
                    break

                if not member.isfile():
                    continue

                normalized_path = (
                    member.name
                    .replace("\\", "/")
                    .strip("/")
                )

                parts = normalized_path.split("/")

                # Valid Enron email:
                # maildir/<user>/<folder>/<message>
                if len(parts) < 4:
                    continue

                if parts[0].lower() != "maildir":
                    continue

                if not parts[-1]:
                    continue

                extracted = tar.extractfile(member)

                if extracted is None:
                    continue

                raw_bytes = extracted.read()

                if not raw_bytes.strip():
                    continue

                try:
                    record = EnronNormalizer.normalize(
                        raw_bytes=raw_bytes,
                        source_path=member.name,
                    )

                    graph, trajectory = (
                        EnronMemoryTrajectoryService
                        .build_from_record(record)
                    )

                except Exception as exc:
                    raise RuntimeError(
                        "Failed while processing Enron member: "
                        f"{member.name}"
                    ) from exc

                trajectory_id = (
                    f"enron-trajectory-"
                    f"{trajectories_processed + 1}"
                )

                for step_number, object_id in enumerate(
                    trajectory,
                    start=1,
                ):

                    data_object = graph.get_object(
                        object_id
                    )

                    if data_object is None:
                        raise RuntimeError(
                            "Trajectory references missing "
                            f"DataObject: {object_id}"
                        )

                    parents = graph.get_parents(
                        object_id
                    )

                    record_output = {
                        "trajectory_id": trajectory_id,
                        "scenario_id": cls.SCENARIO_ID,
                        "scenario_type": cls.SCENARIO_TYPE,
                        "step_number": step_number,
                        "total_steps": len(trajectory),
                        "tool_name": cls.TOOL_NAMES[
                            step_number
                        ],
                        "input_object_ids": parents,
                        "output_object_id": object_id,
                        "label": cls.STEP_LABELS[
                            step_number
                        ],
                        "provenance_depth": graph.get_depth(
                            object_id
                        ),
                        "transformation_depth": (
                            graph.get_transformation_depth(
                                object_id
                            )
                        ),
                        "source": data_object.source,
                        "sensitivity": data_object.sensitivity,
                        "transformation": (
                            data_object.transformation
                        ),
                        "content_hash": data_object.content_hash,
                    }

                    file.write(
                        json.dumps(
                            record_output,
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    total_records += 1

                trajectories_processed += 1

                print(
                    f"[{trajectories_processed:3d}/"
                    f"{max_records}] "
                    f"trajectory generated "
                    f"({total_records} records)"
                )

        if trajectories_processed == 0:
            raise RuntimeError(
                "No usable Enron email members were found."
            )

        expected_records = (
            trajectories_processed * 6
        )

        if total_records != expected_records:
            raise RuntimeError(
                "Unexpected record count: "
                f"generated {total_records}, "
                f"expected {expected_records}."
            )

        return total_records