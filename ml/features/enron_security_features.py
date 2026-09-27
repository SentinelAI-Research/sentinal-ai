from __future__ import annotations

import json
from pathlib import Path


class EnronSecurityFeatureBuilder:
    """
    Converts Enron security trajectory records into
    model-ready numeric features.
    """

    TOOL_ENCODING = {
        "read_file": 0,
        "summarize": 1,
        "write_memory": 2,
        "read_memory": 3,
        "paraphrase": 4,
        "send_message": 5,
    }

    LABEL_ENCODING = {
        "SAFE": 0,
        "EARLY_RISK": 1,
        "VIOLATION": 2,
    }

    @classmethod
    def build_record_features(
        cls,
        record: dict,
    ) -> dict:
        step_number = int(record["step_number"])
        total_steps = int(record["total_steps"])
        provenance_depth = int(
            record["provenance_depth"]
        )
        transformation_depth = int(
            record["transformation_depth"]
        )

        tool_name = record["tool_name"]
        label = record["label"]

        if total_steps <= 0:
            raise ValueError(
                "total_steps must be greater than zero."
            )

        if tool_name not in cls.TOOL_ENCODING:
            raise ValueError(
                f"Unknown tool: {tool_name}"
            )

        if label not in cls.LABEL_ENCODING:
            raise ValueError(
                f"Unknown label: {label}"
            )

        return {
            "trajectory_id": record["trajectory_id"],
            "scenario_id": record["scenario_id"],
            "step_number": step_number,
            "total_steps": total_steps,
            "trajectory_progress": (
                step_number / total_steps
            ),
            "provenance_depth": provenance_depth,
            "transformation_depth": transformation_depth,
            "tool_name": tool_name,
            "tool_encoded": cls.TOOL_ENCODING[
                tool_name
            ],
            "is_memory_operation": int(
                tool_name
                in {
                    "write_memory",
                    "read_memory",
                }
            ),
            "is_external_action": int(
                tool_name
                in {
                    "send_message",
                }
            ),
            "is_transformation": int(
                tool_name
                in {
                    "summarize",
                    "paraphrase",
                }
            ),
            "has_input_object": int(
                len(record["input_object_ids"]) > 0
            ),
            "sensitivity": record["sensitivity"],
            "transformation": record[
                "transformation"
            ],
            "label": label,
            "label_encoded": cls.LABEL_ENCODING[
                label
            ],
        }

    @classmethod
    def build_file(
        cls,
        input_path: str | Path,
        output_path: str | Path,
    ) -> int:
        input_file = Path(input_path)
        output_file = Path(output_path)

        if not input_file.exists():
            raise FileNotFoundError(
                f"Input dataset not found: {input_file}"
            )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        count = 0

        with (
            input_file.open(
                "r",
                encoding="utf-8",
            ) as source,
            output_file.open(
                "w",
                encoding="utf-8",
            ) as destination,
        ):
            for line in source:

                if not line.strip():
                    continue

                record = json.loads(line)

                features = cls.build_record_features(
                    record
                )

                destination.write(
                    json.dumps(features)
                    + "\n"
                )

                count += 1

        return count