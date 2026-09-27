"""
SENTINELAI — Step 58
Prepare ML features for the provenance hard-negative experiment.

Creates two aligned feature views:

1. WITHOUT PROVENANCE
2. WITH PROVENANCE

The two views contain exactly the same records and trajectory IDs.
The only intentional difference is the availability of provenance-derived
features.

This script does not train a model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path("data/processed/hard_negative")

INPUT_FILE = (
    BASE_DIR / "provenance_hard_negative_experiment.jsonl"
)

OUTPUT_WITHOUT = (
    BASE_DIR
    / "provenance_hard_negative_features_without_provenance.jsonl"
)

OUTPUT_WITH = (
    BASE_DIR
    / "provenance_hard_negative_features_with_provenance.jsonl"
)


# ---------------------------------------------------------------------------
# Feature encoding
# ---------------------------------------------------------------------------

TOOL_ENCODING = {
    "read_file": 0,
    "summarize": 1,
    "write_memory": 2,
    "read_memory": 3,
    "paraphrase": 4,
    "send_message": 5,
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

ANCESTOR_SENSITIVITY_ENCODING = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
    "RESTRICTED": 3,
}


# ---------------------------------------------------------------------------
# Fields that must never become model features
# ---------------------------------------------------------------------------

IDENTIFIER_FIELDS = {
    "trajectory_id",
    "original_trajectory_id",
    "source_record_id",
    "scenario_id",
    "scenario_type",
    "variant",
    "object_id",
    "parent_object_id",
    "lineage_object_ids",
    "content_hash",
    "content",
}

TARGET_FIELDS = {
    "label",
    "label_encoded",
}

PROVENANCE_FIELDS = {
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity",
}


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open(
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
                    f"Invalid JSON at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at line "
                    f"{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}"
        )

    return records


def write_jsonl(
    path: Path,
    records: list[dict[str, Any]],
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


# ---------------------------------------------------------------------------
# Required field validation
# ---------------------------------------------------------------------------

def validate_required_fields(
    records: list[dict[str, Any]],
) -> None:

    required = {
        "trajectory_id",
        "source_dataset",
        "step_number",
        "total_steps",
        "trajectory_progress",
        "tool_name",
        "transformation",
        "sensitivity",
        "label",
        "label_encoded",
    }

    for index, record in enumerate(records):

        missing = required - set(
            record.keys()
        )

        if missing:
            raise ValueError(
                f"Record {index} is missing fields: "
                f"{sorted(missing)}"
            )


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

def encode_tool(
    value: str,
) -> int:

    if value not in TOOL_ENCODING:
        raise ValueError(
            f"Unknown tool_name: {value}"
        )

    return TOOL_ENCODING[value]


def encode_transformation(
    value: str,
) -> int:

    if value not in TRANSFORMATION_ENCODING:
        raise ValueError(
            f"Unknown transformation: {value}"
        )

    return TRANSFORMATION_ENCODING[value]


def encode_sensitivity(
    value: str,
) -> int:

    if value not in SENSITIVITY_ENCODING:
        raise ValueError(
            f"Unknown sensitivity: {value}"
        )

    return SENSITIVITY_ENCODING[value]


def encode_ancestor_sensitivity(
    value: str,
) -> int:

    if value not in ANCESTOR_SENSITIVITY_ENCODING:
        raise ValueError(
            f"Unknown ancestor sensitivity: {value}"
        )

    return ANCESTOR_SENSITIVITY_ENCODING[value]


# ---------------------------------------------------------------------------
# Base feature construction
# ---------------------------------------------------------------------------

def build_base_features(
    record: dict[str, Any],
) -> dict[str, Any]:

    tool_name = str(
        record["tool_name"]
    )

    transformation = str(
        record["transformation"]
    )

    sensitivity = str(
        record["sensitivity"]
    )

    features: dict[str, Any] = {
        # Dataset/domain information.
        "source_dataset": str(
            record["source_dataset"]
        ),

        # We retain trajectory ID in the feature file for grouping,
        # leakage checks and traceability, but the training script must
        # exclude it from X.
        "trajectory_id": str(
            record["trajectory_id"]
        ),

        "step_number": int(
            record["step_number"]
        ),

        "total_steps": int(
            record["total_steps"]
        ),

        "trajectory_progress": float(
            record["trajectory_progress"]
        ),

        # Current action.
        "tool_name": tool_name,

        "tool_encoded": encode_tool(
            tool_name
        ),

        # Current transformation.
        "transformation": transformation,

        "transformation_encoded": (
            encode_transformation(
                transformation
            )
        ),

        # Current sensitivity.
        "sensitivity": sensitivity,

        "sensitivity_encoded": (
            encode_sensitivity(
                sensitivity
            )
        ),

        # Action properties.
        "has_input_object": int(
            record.get(
                "has_input_object",
                0,
            )
        ),

        "is_memory_operation": int(
            record.get(
                "is_memory_operation",
                0,
            )
        ),

        "is_transformation": int(
            record.get(
                "is_transformation",
                0,
            )
        ),

        "is_external_action": int(
            record.get(
                "is_external_action",
                0,
            )
        ),

        # Ground-truth label.
        #
        # This remains in the JSONL for supervised learning but is excluded
        # from the actual model feature matrix.
        "label": str(
            record["label"]
        ),

        "label_encoded": int(
            record["label_encoded"]
        ),
    }

    return features


# ---------------------------------------------------------------------------
# Provenance feature construction
# ---------------------------------------------------------------------------

def add_provenance_features(
    features: dict[str, Any],
    record: dict[str, Any],
) -> None:

    features["provenance_depth"] = int(
        record.get(
            "provenance_depth",
            0,
        )
    )

    features["transformation_depth"] = int(
        record.get(
            "transformation_depth",
            0,
        )
    )

    features["has_provenance"] = int(
        record.get(
            "has_provenance",
            0,
        )
    )

    features["has_sensitive_ancestor"] = int(
        record.get(
            "has_sensitive_ancestor",
            0,
        )
    )

    features["sensitive_ancestor_count"] = int(
        record.get(
            "sensitive_ancestor_count",
            0,
        )
    )

    ancestor_sensitivity = str(
        record.get(
            "max_ancestor_sensitivity",
            "PUBLIC",
        )
    )

    features[
        "max_ancestor_sensitivity_encoded"
    ] = encode_ancestor_sensitivity(
        ancestor_sensitivity
    )


# ---------------------------------------------------------------------------
# Build both feature views
# ---------------------------------------------------------------------------

def prepare_feature_views(
    records: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:

    without_provenance: list[
        dict[str, Any]
    ] = []

    with_provenance: list[
        dict[str, Any]
    ] = []

    for record in records:

        base_features = build_base_features(
            record
        )

        without_record = dict(
            base_features
        )

        with_record = dict(
            base_features
        )

        add_provenance_features(
            with_record,
            record,
        )

        without_provenance.append(
            without_record
        )

        with_provenance.append(
            with_record
        )

    return (
        without_provenance,
        with_provenance,
    )


# ---------------------------------------------------------------------------
# Alignment validation
# ---------------------------------------------------------------------------

def get_alignment_key(
    record: dict[str, Any],
) -> tuple[str, int]:

    return (
        str(record["trajectory_id"]),
        int(record["step_number"]),
    )


def validate_alignment(
    without_provenance: list[dict[str, Any]],
    with_provenance: list[dict[str, Any]],
) -> None:

    if len(without_provenance) != len(
        with_provenance
    ):
        raise ValueError(
            "Feature views have different record counts."
        )

    without_keys = {
        get_alignment_key(record)
        for record in without_provenance
    }

    with_keys = {
        get_alignment_key(record)
        for record in with_provenance
    }

    if without_keys != with_keys:
        raise ValueError(
            "Feature views are not aligned."
        )

    print(
        "✓ Both feature views contain the same "
        "trajectory/step records."
    )


def validate_provenance_difference(
    without_provenance: list[dict[str, Any]],
    with_provenance: list[dict[str, Any]],
) -> None:

    without_keys = set(
        without_provenance[0].keys()
    )

    with_keys = set(
        with_provenance[0].keys()
    )

    unexpected_in_without = (
        without_keys & PROVENANCE_FIELDS
    )

    if unexpected_in_without:
        raise ValueError(
            "Provenance features leaked into the "
            "without-provenance view: "
            f"{sorted(unexpected_in_without)}"
        )

    required_provenance = {
        "provenance_depth",
        "transformation_depth",
        "has_provenance",
        "has_sensitive_ancestor",
        "sensitive_ancestor_count",
        "max_ancestor_sensitivity_encoded",
    }

    missing = required_provenance - with_keys

    if missing:
        raise ValueError(
            "Missing required provenance features: "
            f"{sorted(missing)}"
        )

    print(
        "✓ Without-provenance view contains no "
        "provenance-derived features."
    )

    print(
        "✓ With-provenance view contains all "
        "required provenance features."
    )


# ---------------------------------------------------------------------------
# Leakage validation
# ---------------------------------------------------------------------------

def validate_no_label_features(
    records: list[dict[str, Any]],
) -> None:

    for record in records:

        for field in TARGET_FIELDS:

            if field not in record:
                raise ValueError(
                    f"Expected target field '{field}' "
                    "to remain available."
                )

    print(
        "✓ Target labels are retained for supervised "
        "learning and will be excluded during training."
    )


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_statistics(
    name: str,
    records: list[dict[str, Any]],
) -> None:

    from collections import Counter

    trajectories = {
        str(record["trajectory_id"])
        for record in records
    }

    sources = Counter(
        record["source_dataset"]
        for record in records
    )

    variants = Counter(
        record.get(
            "variant",
            "unknown",
        )
        for record in records
    )

    labels = Counter(
        record["label"]
        for record in records
    )

    print(
        f"\n{name}"
    )

    print(
        "-" * 60
    )

    print(
        f"Records:       {len(records)}"
    )

    print(
        f"Trajectories:  {len(trajectories)}"
    )

    print(
        "Sources:"
    )

    for source, count in sorted(
        sources.items()
    ):
        print(
            f"  {source}: {count}"
        )

    print(
        "Variants:"
    )

    for variant, count in sorted(
        variants.items()
    ):
        print(
            f"  {variant}: {count}"
        )

    print(
        "Labels:"
    )

    for label, count in sorted(
        labels.items()
    ):
        print(
            f"  {label}: {count}"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "SENTINELAI — STEP 58"
    )
    print(
        "PROVENANCE HARD-NEGATIVE ML FEATURES"
    )
    print("=" * 70)

    print(
        "\nLoading hard-negative dataset..."
    )

    records = load_jsonl(
        INPUT_FILE
    )

    print(
        f"Loaded records: {len(records)}"
    )

    validate_required_fields(
        records
    )

    validate_no_label_features(
        records
    )

    print(
        "\nBuilding feature views..."
    )

    (
        without_provenance,
        with_provenance,
    ) = prepare_feature_views(
        records
    )

    print(
        f"Without-provenance records: "
        f"{len(without_provenance)}"
    )

    print(
        f"With-provenance records: "
        f"{len(with_provenance)}"
    )

    validate_alignment(
        without_provenance,
        with_provenance,
    )

    validate_provenance_difference(
        without_provenance,
        with_provenance,
    )

    print_statistics(
        "WITHOUT PROVENANCE",
        without_provenance,
    )

    print_statistics(
        "WITH PROVENANCE",
        with_provenance,
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    write_jsonl(
        OUTPUT_WITHOUT,
        without_provenance,
    )

    write_jsonl(
        OUTPUT_WITH,
        with_provenance,
    )

    print(
        "\nSaved feature files:"
    )

    print(
        f"  ✓ {OUTPUT_WITHOUT}"
    )

    print(
        f"  ✓ {OUTPUT_WITH}"
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    without_features = set(
        without_provenance[0].keys()
    )

    with_features = set(
        with_provenance[0].keys()
    )

    provenance_only = sorted(
        with_features - without_features
    )

    print(
        "\nProvenance-only features:"
    )

    for feature in provenance_only:
        print(
            f"  + {feature}"
        )

    print("\n" + "=" * 70)
    print(
        "✓ STEP 58 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()