from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_experiment.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "enron"
)

NON_PROVENANCE_OUTPUT = (
    OUTPUT_DIR
    / "enron_ml_features_without_provenance.jsonl"
)

PROVENANCE_OUTPUT = (
    OUTPUT_DIR
    / "enron_ml_features_with_provenance.jsonl"
)


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


TRANSFORMATION_ENCODING = {
    "source_ingestion": 0,
    "summarize": 1,
    "write_memory": 2,
    "read_memory": 3,
    "paraphrase": 4,
    "send_message": 5,
}


SENSITIVITY_ENCODING = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
    "RESTRICTED": 3,
}


def encode_tool(tool_name: str) -> int:
    if tool_name not in TOOL_ENCODING:
        raise ValueError(
            f"Unknown tool name: {tool_name}"
        )

    return TOOL_ENCODING[tool_name]


def encode_label(label: str) -> int:
    if label not in LABEL_ENCODING:
        raise ValueError(
            f"Unknown label: {label}"
        )

    return LABEL_ENCODING[label]


def encode_transformation(transformation: str) -> int:
    if transformation not in TRANSFORMATION_ENCODING:
        raise ValueError(
            f"Unknown transformation: {transformation}"
        )

    return TRANSFORMATION_ENCODING[transformation]


def encode_sensitivity(sensitivity: str) -> int:
    if sensitivity not in SENSITIVITY_ENCODING:
        raise ValueError(
            f"Unknown sensitivity: {sensitivity}"
        )

    return SENSITIVITY_ENCODING[sensitivity]


def build_features(record: dict) -> tuple[dict, dict]:
    """
    Build two feature views from exactly the same trajectory record.

    The first view intentionally excludes provenance-derived features.

    The second view contains the same base features plus provenance
    information.

    No label-derived feature is included in either view.
    """

    step_number = int(record["step_number"])
    total_steps = int(record["total_steps"])

    tool_name = record["tool_name"]
    transformation = record["transformation"]
    sensitivity = record["sensitivity"]

    trajectory_progress = (
        step_number / total_steps
        if total_steps > 0
        else 0.0
    )

    is_memory_operation = int(
        tool_name in {
            "write_memory",
            "read_memory",
        }
    )

    is_external_action = int(
        tool_name == "send_message"
    )

    is_transformation = int(
        tool_name in {
            "summarize",
            "paraphrase",
        }
    )

    has_input_object = int(
        len(record["input_object_ids"]) > 0
    )

    base_features = {
        "trajectory_id": record["trajectory_id"],
        "scenario_id": record["scenario_id"],
        "step_number": step_number,
        "total_steps": total_steps,
        "trajectory_progress": trajectory_progress,

        "tool_name": tool_name,
        "tool_encoded": encode_tool(tool_name),

        "transformation": transformation,
        "transformation_encoded": encode_transformation(
            transformation
        ),

        "sensitivity": sensitivity,
        "sensitivity_encoded": encode_sensitivity(
            sensitivity
        ),

        "is_memory_operation": is_memory_operation,
        "is_external_action": is_external_action,
        "is_transformation": is_transformation,
        "has_input_object": has_input_object,

        "label": record["label"],
        "label_encoded": encode_label(record["label"]),
    }

    provenance_features = {
        **base_features,

        "provenance_depth": int(
            record["provenance_depth"]
        ),

        "transformation_depth": int(
            record["transformation_depth"]
        ),

        "has_provenance": int(
            int(record["provenance_depth"]) > 0
        ),
    }

    return base_features, provenance_features


def load_records() -> list[dict]:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Input dataset not found: {INPUT_PATH}"
        )

    records: list[dict] = []

    with INPUT_PATH.open(
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
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at line {line_number}"
                ) from exc

            records.append(record)

    return records


def write_jsonl(
    path: Path,
    records: list[dict],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                )
                + "\n"
            )


def validate_feature_pair(
    without_provenance: list[dict],
    with_provenance: list[dict],
) -> None:
    if len(without_provenance) != len(with_provenance):
        raise RuntimeError(
            "Feature datasets contain different numbers of records."
        )

    for index, (
        base_record,
        provenance_record,
    ) in enumerate(
        zip(
            without_provenance,
            with_provenance,
        ),
        start=1,
    ):
        identity_fields = [
            "trajectory_id",
            "scenario_id",
            "step_number",
            "total_steps",
            "tool_name",
            "tool_encoded",
            "transformation",
            "transformation_encoded",
            "sensitivity",
            "sensitivity_encoded",
            "is_memory_operation",
            "is_external_action",
            "is_transformation",
            "has_input_object",
            "label",
            "label_encoded",
        ]

        for field in identity_fields:
            if base_record[field] != provenance_record[field]:
                raise RuntimeError(
                    f"Feature mismatch at record {index}, "
                    f"field '{field}'."
                )

        if "provenance_depth" in base_record:
            raise RuntimeError(
                "Non-provenance feature set contains "
                "provenance_depth."
            )

        if "transformation_depth" in base_record:
            raise RuntimeError(
                "Non-provenance feature set contains "
                "transformation_depth."
            )

        if "has_provenance" in base_record:
            raise RuntimeError(
                "Non-provenance feature set contains "
                "has_provenance."
            )

        if "provenance_depth" not in provenance_record:
            raise RuntimeError(
                f"Record {index} is missing provenance_depth."
            )

        if "transformation_depth" not in provenance_record:
            raise RuntimeError(
                f"Record {index} is missing transformation_depth."
            )

        if "has_provenance" not in provenance_record:
            raise RuntimeError(
                f"Record {index} is missing has_provenance."
            )


def main() -> None:
    print("=" * 70)
    print("SENTINELAI — ENRON ML FEATURE PREPARATION")
    print("=" * 70)

    print(f"Input dataset:")
    print(INPUT_PATH)
    print()

    records = load_records()

    print(
        f"Loaded {len(records)} experiment records."
    )

    if not records:
        raise RuntimeError(
            "No records were loaded."
        )

    without_provenance: list[dict] = []
    with_provenance: list[dict] = []

    for record in records:
        base_features, provenance_features = (
            build_features(record)
        )

        without_provenance.append(
            base_features
        )

        with_provenance.append(
            provenance_features
        )

    validate_feature_pair(
        without_provenance,
        with_provenance,
    )

    write_jsonl(
        NON_PROVENANCE_OUTPUT,
        without_provenance,
    )

    write_jsonl(
        PROVENANCE_OUTPUT,
        with_provenance,
    )

    print()
    print(
        f"✓ Without-provenance features: "
        f"{len(without_provenance)} records"
    )

    print(
        f"✓ With-provenance features: "
        f"{len(with_provenance)} records"
    )

    print()
    print(
        f"✓ Written:"
    )
    print(
        f"  {NON_PROVENANCE_OUTPUT}"
    )
    print(
        f"  {PROVENANCE_OUTPUT}"
    )

    print()
    print(
        "✓ Both datasets contain identical examples "
        "and labels."
    )

    print(
        "✓ The without-provenance dataset contains "
        "no provenance-derived features."
    )

    print(
        "✓ The provenance-aware dataset contains "
        "provenance depth and transformation depth."
    )

    print()
    print("=" * 70)
    print("STEP 51 FEATURE PREPARATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()