from __future__ import annotations

import json
from pathlib import Path


class AMITransformationFeatureBuilder:
    """
    Converts validated AMI transformation records into
    reproducible ML feature records.

    These features describe the observed transformation
    and provenance structure. They do not contain security
    labels yet.
    """

    @staticmethod
    def build_record_features(record: dict) -> dict:
        source_object = record["source_object"]
        transformed_object = record["transformed_object"]
        provenance = record["provenance"]
        transformation = record["transformation"]

        source_length = int(transformation["source_text_length"])

        reference_length = int(transformation["reference_summary_length"])

        compression_ratio = (
            reference_length / source_length if source_length > 0 else 0.0
        )

        return {
            "record_id": record["record_id"],
            "source_dataset": record["dataset"],
            "source_text_length": source_length,
            "reference_summary_length": reference_length,
            "compression_ratio": compression_ratio,
            "provenance_depth": int(provenance["depth"]),
            "transformation_depth": int(provenance["transformation_depth"]),
            "has_parent": int(len(transformed_object["parents"]) > 0),
            "source_sensitivity": source_object["sensitivity"],
            "transformation_name": transformed_object["transformation"],
            "source_content_hash": source_object["content_hash"],
            "output_content_hash": transformed_object["content_hash"],
        }

    @classmethod
    def build_file(
        cls,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Input dataset does not exist: " f"{input_path}")

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = 0

        with input_path.open(
            "r",
            encoding="utf-8",
        ) as input_file, output_path.open(
            "w",
            encoding="utf-8",
        ) as output_file:

            for line_number, line in enumerate(
                input_file,
                start=1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line " f"{line_number}: {exc}"
                    ) from exc

                features = cls.build_record_features(record)

                output_file.write(
                    json.dumps(
                        features,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                count += 1

        return count
