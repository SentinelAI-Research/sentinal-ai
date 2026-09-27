import json
from pathlib import Path

from ml.data.ami_normalizer import AMINormalizer
from ml.data.ami_provenance import AMIProvenanceLoader
from ml.data.ami_transformation import AMITransformationService


class AMITransformationDatasetBuilder:
    """
    Builds a reproducible transformation dataset from unique
    real AMI records.

    The builder deliberately deduplicates AMI records by the
    dataset's own record ID before creating DataObjects and
    transformations.
    """

    def __init__(
        self,
        input_path: str | Path,
        output_path: str | Path,
    ) -> None:
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)

    def _load_unique_records(
        self,
        max_records: int,
    ) -> list:
        if not self.input_path.exists():
            raise FileNotFoundError(
                f"AMI input file does not exist: " f"{self.input_path}"
            )

        records = []
        seen_ids = set()

        with self.input_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    raw_record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line " f"{line_number}: {exc}"
                    ) from exc

                normalized = AMINormalizer.normalize_record(raw_record)

                if normalized.record_id in seen_ids:
                    continue

                seen_ids.add(normalized.record_id)

                records.append(normalized)

                if len(records) >= max_records:
                    break

        return records

    def build(
        self,
        max_records: int = 100,
    ) -> int:
        if max_records <= 0:
            raise ValueError("max_records must be greater than zero.")

        self.output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        normalized_records = self._load_unique_records(
            max_records=max_records,
        )

        if not normalized_records:
            raise ValueError("No valid unique AMI records found.")

        # Build DataObjects and provenance from the
        # same unique normalized records.
        source_objects = []

        from backend.dataobjects.factory import (
            create_data_object,
        )
        from backend.provenance.graph import (
            ProvenanceGraph,
        )

        graph = ProvenanceGraph()

        for record in normalized_records:
            object_id = f"ami-train-{record.record_id}"

            source_object = create_data_object(
                object_id=object_id,
                source=(f"AMI Meeting Corpus:" f"{record.record_id}"),
                content=record.source_text,
                sensitivity="PUBLIC",
                transformation="source_ingestion",
                parents=[],
                created_step="ami-ingestion-train",
            )

            graph.add_object(source_object)

            source_objects.append(source_object)

        written_records = 0

        with self.output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for record, source_object in zip(
                normalized_records,
                source_objects,
            ):
                step_id = f"ami-summary-{record.record_id}"

                summary_object = AMITransformationService.summarize_object(
                    graph=graph,
                    source_object=source_object,
                    source_text=record.source_text,
                    step_id=step_id,
                )

                output_record = {
                    "record_id": record.record_id,
                    "dataset": "AMI Meeting Corpus",
                    "source_object": {
                        "object_id": source_object.object_id,
                        "source": source_object.source,
                        "sensitivity": source_object.sensitivity,
                        "content_hash": (source_object.content_hash),
                        "transformation": (source_object.transformation),
                        "created_step": (source_object.created_step),
                    },
                    "transformed_object": {
                        "object_id": (summary_object.object_id),
                        "source": summary_object.source,
                        "sensitivity": (summary_object.sensitivity),
                        "content_hash": (summary_object.content_hash),
                        "transformation": (summary_object.transformation),
                        "parents": (summary_object.parents),
                        "created_step": (summary_object.created_step),
                    },
                    "provenance": {
                        "parent_object_id": (source_object.object_id),
                        "child_object_id": (summary_object.object_id),
                        "depth": graph.get_depth(summary_object.object_id),
                        "transformation_depth": (
                            graph.get_transformation_depth(summary_object.object_id)
                        ),
                    },
                    "transformation": {
                        "name": "summarize",
                        "source_text_length": len(record.source_text),
                        "reference_summary_length": len(record.reference_text),
                        "output_hash": (summary_object.content_hash),
                    },
                }

                output_file.write(
                    json.dumps(
                        output_record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                written_records += 1

        return written_records
