"""
SENTINELAI — Step 57
Build provenance-sensitive hard-negative trajectories.

Purpose
-------
The original controlled dataset produced perfect ML scores because the
trajectory action pattern directly encoded the labels.

This script creates paired trajectories where the final/current observation
can be identical, while the provenance lineage differs.

Scenario:

BENIGN:
    PUBLIC source
        -> summarize
        -> write_memory
        -> read_memory
        -> paraphrase
        -> send_message
        -> SAFE

RISKY:
    CONFIDENTIAL source
        -> summarize
        -> write_memory
        -> read_memory
        -> paraphrase
        -> send_message
        -> VIOLATION

The risky trajectory contains a confidential ancestor even though the
current/final sensitivity is normalized to PUBLIC.

This is a CONTROLLED provenance experiment. The underlying AMI/Enron
content remains public-source material; CONFIDENTIAL/INTERNAL labels are
experimental security metadata, not claims about the original datasets'
real-world confidentiality.

The generated records include explicit provenance lineage metadata so that
later feature engineering can compare:

    action-only information
vs.
    provenance-aware information
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path("data/processed")

AMI_INPUT = BASE_DIR / "ami" / "ami_security_experiment.jsonl"
ENRON_INPUT = BASE_DIR / "enron" / "enron_security_experiment.jsonl"

OUTPUT_DIR = BASE_DIR / "hard_negative"

OUTPUT_FILE = (
    OUTPUT_DIR / "provenance_hard_negative_experiment.jsonl"
)

SUMMARY_FILE = (
    OUTPUT_DIR / "provenance_hard_negative_summary.json"
)

TARGET_TRAJECTORIES_PER_DATASET = 100

STEPS_PER_TRAJECTORY = 6


# ---------------------------------------------------------------------------
# Controlled scenario
# ---------------------------------------------------------------------------

TOOLS = [
    "read_file",
    "summarize",
    "write_memory",
    "read_memory",
    "paraphrase",
    "send_message",
]

TRANSFORMATIONS = [
    "source_ingestion",
    "summarize",
    "write_memory",
    "read_memory",
    "paraphrase",
    "send_message",
]

# The current object is intentionally PUBLIC at step 6 in both variants.
CURRENT_SENSITIVITY = "PUBLIC"

# Experimental lineage sensitivity.
BENIGN_SOURCE_SENSITIVITY = "PUBLIC"
RISKY_SOURCE_SENSITIVITY = "CONFIDENTIAL"


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(
    path: Path,
) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input file does not exist: {path}"
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
                    f"Invalid JSON in {path} at line "
                    f"{line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
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
# Record helpers
# ---------------------------------------------------------------------------

def stable_hash(
    value: str,
) -> str:

    return hashlib.sha256(
        value.encode("utf-8")
    ).hexdigest()


def trajectory_records(
    records: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:

    grouped: dict[
        str,
        list[dict[str, Any]]
    ] = {}

    for record in records:

        trajectory_id = str(
            record["trajectory_id"]
        )

        grouped.setdefault(
            trajectory_id,
            [],
        ).append(record)

    for trajectory_id in grouped:

        grouped[trajectory_id].sort(
            key=lambda item: int(
                item["step_number"]
            )
        )

    return grouped


def get_first_record(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    if not records:
        raise ValueError(
            "Trajectory contains no records."
        )

    return records[0]


def get_source_dataset(
    record: dict[str, Any],
) -> str:

    source = record.get(
        "source_dataset"
    )

    if source:
        return str(source)

    trajectory_id = str(
        record["trajectory_id"]
    ).lower()

    if trajectory_id.startswith("ami"):
        return "ami"

    if trajectory_id.startswith("enron"):
        return "enron"

    raise ValueError(
        "Unable to determine source dataset for "
        f"trajectory {trajectory_id}."
    )


# ---------------------------------------------------------------------------
# Provenance construction
# ---------------------------------------------------------------------------

def build_object_id(
    trajectory_id: str,
    variant: str,
    step_number: int,
) -> str:

    return (
        f"{trajectory_id}"
        f"-{variant}"
        f"-step-{step_number}"
    )


def build_parent_id(
    trajectory_id: str,
    variant: str,
    step_number: int,
) -> str | None:

    if step_number <= 1:
        return None

    return build_object_id(
        trajectory_id,
        variant,
        step_number - 1,
    )


def build_lineage_ids(
    trajectory_id: str,
    variant: str,
    step_number: int,
) -> list[str]:

    return [
        build_object_id(
            trajectory_id,
            variant,
            step,
        )
        for step in range(
            1,
            step_number + 1,
        )
    ]


# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------

def get_content(
    record: dict[str, Any],
) -> str:

    for key in (
        "content",
        "source_text",
        "text",
        "output",
        "value",
    ):

        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value.strip()

    # The existing security experiment stores a content hash in some
    # stages. We do not fabricate source text.
    # The original record remains the source of truth.
    content_hash = record.get(
        "content_hash"
    )

    if content_hash:
        return f"[source-content-hash:{content_hash}]"

    return ""


# ---------------------------------------------------------------------------
# Build one paired trajectory
# ---------------------------------------------------------------------------

def build_variant_trajectory(
    source_records: list[dict[str, Any]],
    variant: str,
    source_sensitivity: str,
    label_scheme: str,
) -> list[dict[str, Any]]:

    first = get_first_record(
        source_records
    )

    original_trajectory_id = str(
        first["trajectory_id"]
    )

    source_dataset = get_source_dataset(
        first
    )

    output_records: list[
        dict[str, Any]
    ] = []

    new_trajectory_id = (
        f"pn-{source_dataset}-"
        f"{original_trajectory_id}-"
        f"{variant}"
    )

    for step_number in range(
        1,
        STEPS_PER_TRAJECTORY + 1,
    ):

        original = source_records[
            step_number - 1
        ]

        object_id = build_object_id(
            new_trajectory_id,
            variant,
            step_number,
        )

        parent_id = build_parent_id(
            new_trajectory_id,
            variant,
            step_number,
        )

        lineage_ids = build_lineage_ids(
            new_trajectory_id,
            variant,
            step_number,
        )

        # ---------------------------------------------------------------
        # Controlled labels
        # ---------------------------------------------------------------

        if step_number == 1:

            label = "SAFE"
            label_encoded = 0

        elif step_number in (
            2,
            3,
            4,
            5,
        ):

            label = "EARLY_RISK"
            label_encoded = 1

        else:

            if label_scheme == "BENIGN":
                label = "SAFE"
                label_encoded = 0

            elif label_scheme == "VIOLATION":
                label = "VIOLATION"
                label_encoded = 2

            else:
                raise ValueError(
                    f"Unknown label scheme: {label_scheme}"
                )

        # ---------------------------------------------------------------
        # Controlled current sensitivity
        # ---------------------------------------------------------------

        # Both variants deliberately have PUBLIC current sensitivity.
        #
        # This is the key hard-negative property:
        #
        #     current observation = same
        #     provenance lineage = different
        #
        current_sensitivity = CURRENT_SENSITIVITY

        # At step 1, the source retains its actual experimental
        # source sensitivity.
        if step_number == 1:
            current_sensitivity = source_sensitivity

        # ---------------------------------------------------------------
        # Provenance metadata
        # ---------------------------------------------------------------

        has_sensitive_ancestor = (
            source_sensitivity == "CONFIDENTIAL"
        )

        sensitive_ancestor_count = (
            step_number
            if has_sensitive_ancestor
            else 0
        )

        max_ancestor_sensitivity = (
            "CONFIDENTIAL"
            if has_sensitive_ancestor
            else "PUBLIC"
        )

        # At step 1 the object is its own source.
        # At later steps the full ancestor chain is retained.
        provenance_depth = step_number - 1

        transformation_depth = max(
            0,
            step_number - 1,
        )

        # ---------------------------------------------------------------
        # Content
        # ---------------------------------------------------------------

        content = get_content(
            original
        )

        content_hash = stable_hash(
            f"{new_trajectory_id}|"
            f"{step_number}|"
            f"{content}"
        )

        # ---------------------------------------------------------------
        # Record
        # ---------------------------------------------------------------

        record: dict[str, Any] = {
            "trajectory_id": new_trajectory_id,
            "original_trajectory_id": original_trajectory_id,
            "source_dataset": source_dataset,
            "scenario_id": (
                "provenance-hard-negative"
            ),
            "scenario_type": (
                "PROVENANCE_TAINT_COMPARISON"
            ),
            "variant": variant,
            "step_number": step_number,
            "total_steps": STEPS_PER_TRAJECTORY,
            "trajectory_progress": (
                step_number
                / STEPS_PER_TRAJECTORY
            ),
            "tool_name": TOOLS[
                step_number - 1
            ],
            "transformation": TRANSFORMATIONS[
                step_number - 1
            ],
            "sensitivity": current_sensitivity,
            "source_sensitivity": source_sensitivity,
            "label": label,
            "label_encoded": label_encoded,
            "object_id": object_id,
            "parent_object_id": parent_id,
            "lineage_object_ids": lineage_ids,
            "provenance_depth": provenance_depth,
            "transformation_depth": transformation_depth,
            "has_provenance": int(
                provenance_depth > 0
            ),
            "has_input_object": int(
                step_number > 1
            ),
            "is_memory_operation": int(
                step_number in (3, 4)
            ),
            "is_transformation": int(
                step_number in (2, 5)
            ),
            "is_external_action": int(
                step_number == 6
            ),
            "has_sensitive_ancestor": int(
                has_sensitive_ancestor
            ),
            "sensitive_ancestor_count": (
                sensitive_ancestor_count
            ),
            "max_ancestor_sensitivity": (
                max_ancestor_sensitivity
            ),
            "content_hash": content_hash,
            "content": content,
            "source_record_id": str(
                original.get(
                    "source_record_id",
                    original.get(
                        "record_id",
                        original_trajectory_id,
                    ),
                )
            ),
            "controlled_security_scenario": True,
            "security_scenario_note": (
                "Source material is public. "
                "Sensitivity and security labels are "
                "controlled experimental metadata."
            ),
        }

        output_records.append(
            record
        )

    return output_records


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_dataset(
    records: list[dict[str, Any]],
) -> dict[str, Any]:

    if not records:
        raise ValueError(
            "Generated dataset is empty."
        )

    grouped = trajectory_records(
        records
    )

    expected_steps = list(
        range(
            1,
            STEPS_PER_TRAJECTORY + 1,
        )
    )

    for trajectory_id, trajectory in grouped.items():

        steps = [
            int(record["step_number"])
            for record in trajectory
        ]

        if steps != expected_steps:
            raise ValueError(
                f"Invalid steps for trajectory "
                f"{trajectory_id}: {steps}"
            )

        variants = {
            record["variant"]
            for record in trajectory
        }

        if len(variants) != 1:
            raise ValueError(
                f"Trajectory {trajectory_id} contains "
                f"multiple variants: {variants}"
            )

        current_sensitivities = {
            record["sensitivity"]
            for record in trajectory
            if int(record["step_number"]) == 6
        }

        if current_sensitivities != {
            CURRENT_SENSITIVITY
        }:

            raise ValueError(
                f"Final sensitivity mismatch in "
                f"{trajectory_id}: "
                f"{current_sensitivities}"
            )

    variant_counts = Counter(
        record["variant"]
        for record in records
    )

    label_counts = Counter(
        record["label"]
        for record in records
    )

    source_counts = Counter(
        record["source_dataset"]
        for record in records
    )

    return {
        "records": len(records),
        "trajectories": len(grouped),
        "variant_counts": dict(
            variant_counts
        ),
        "label_counts": dict(
            label_counts
        ),
        "source_counts": dict(
            source_counts
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print(
        "SENTINELAI — STEP 57"
    )
    print(
        "PROVENANCE-SENSITIVE HARD-NEGATIVE DATASET"
    )
    print("=" * 70)

    print("\nLoading existing controlled datasets...")

    ami_records = load_jsonl(
        AMI_INPUT
    )

    enron_records = load_jsonl(
        ENRON_INPUT
    )

    ami_grouped = trajectory_records(
        ami_records
    )

    enron_grouped = trajectory_records(
        enron_records
    )

    print(
        f"AMI trajectories available: "
        f"{len(ami_grouped)}"
    )

    print(
        f"Enron trajectories available: "
        f"{len(enron_grouped)}"
    )

    if len(ami_grouped) < TARGET_TRAJECTORIES_PER_DATASET:
        raise ValueError(
            "Not enough AMI trajectories. "
            f"Required {TARGET_TRAJECTORIES_PER_DATASET}, "
            f"found {len(ami_grouped)}."
        )

    if len(enron_grouped) < TARGET_TRAJECTORIES_PER_DATASET:
        raise ValueError(
            "Not enough Enron trajectories. "
            f"Required {TARGET_TRAJECTORIES_PER_DATASET}, "
            f"found {len(enron_grouped)}."
        )

    # Deterministic ordering ensures reproducibility.
    ami_ids = sorted(
        ami_grouped.keys()
    )[
        :TARGET_TRAJECTORIES_PER_DATASET
    ]

    enron_ids = sorted(
        enron_grouped.keys()
    )[
        :TARGET_TRAJECTORIES_PER_DATASET
    ]

    generated_records: list[
        dict[str, Any]
    ] = []

    # ------------------------------------------------------------------
    # AMI
    # ------------------------------------------------------------------

    print("\nBuilding AMI paired trajectories...")

    for trajectory_id in ami_ids:

        source_records = ami_grouped[
            trajectory_id
        ]

        generated_records.extend(
            build_variant_trajectory(
                source_records=source_records,
                variant="benign",
                source_sensitivity=(
                    BENIGN_SOURCE_SENSITIVITY
                ),
                label_scheme="BENIGN",
            )
        )

        generated_records.extend(
            build_variant_trajectory(
                source_records=source_records,
                variant="provenance_risky",
                source_sensitivity=(
                    RISKY_SOURCE_SENSITIVITY
                ),
                label_scheme="VIOLATION",
            )
        )

    # ------------------------------------------------------------------
    # Enron
    # ------------------------------------------------------------------

    print(
        "Building Enron paired trajectories..."
    )

    for trajectory_id in enron_ids:

        source_records = enron_grouped[
            trajectory_id
        ]

        generated_records.extend(
            build_variant_trajectory(
                source_records=source_records,
                variant="benign",
                source_sensitivity=(
                    BENIGN_SOURCE_SENSITIVITY
                ),
                label_scheme="BENIGN",
            )
        )

        generated_records.extend(
            build_variant_trajectory(
                source_records=source_records,
                variant="provenance_risky",
                source_sensitivity=(
                    RISKY_SOURCE_SENSITIVITY
                ),
                label_scheme="VIOLATION",
            )
        )

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    print("\nValidating generated dataset...")

    summary = validate_dataset(
        generated_records
    )

    print(
        f"✓ Records: {summary['records']}"
    )

    print(
        f"✓ Trajectories: "
        f"{summary['trajectories']}"
    )

    print("\nVariants:")

    for variant, count in sorted(
        summary["variant_counts"].items()
    ):
        print(
            f"  {variant}: {count}"
        )

    print("\nSources:")

    for source, count in sorted(
        summary["source_counts"].items()
    ):
        print(
            f"  {source}: {count}"
        )

    print("\nLabels:")

    for label, count in sorted(
        summary["label_counts"].items()
    ):
        print(
            f"  {label}: {count}"
        )

    # ------------------------------------------------------------------
    # Explicit hard-negative validation
    # ------------------------------------------------------------------

    print(
        "\nChecking provenance hard-negative property..."
    )

    grouped = trajectory_records(
        generated_records
    )

    benign_final = [
        record
        for trajectory in grouped.values()
        if trajectory[0]["variant"] == "benign"
        for record in trajectory
        if int(record["step_number"]) == 6
    ]

    risky_final = [
        record
        for trajectory in grouped.values()
        if trajectory[0]["variant"]
        == "provenance_risky"
        for record in trajectory
        if int(record["step_number"]) == 6
    ]

    if not benign_final:
        raise ValueError(
            "No benign final-step records found."
        )

    if not risky_final:
        raise ValueError(
            "No risky final-step records found."
        )

    benign_sensitivity = {
        record["sensitivity"]
        for record in benign_final
    }

    risky_sensitivity = {
        record["sensitivity"]
        for record in risky_final
    }

    if benign_sensitivity != risky_sensitivity:
        raise ValueError(
            "Hard-negative requirement failed: "
            "final sensitivity differs between variants."
        )

    if benign_sensitivity != {
        CURRENT_SENSITIVITY
    }:
        raise ValueError(
            "Final current sensitivity is not PUBLIC."
        )

    benign_lineage = {
        int(record["has_sensitive_ancestor"])
        for record in benign_final
    }

    risky_lineage = {
        int(record["has_sensitive_ancestor"])
        for record in risky_final
    }

    if benign_lineage != {0}:
        raise ValueError(
            "Benign trajectories unexpectedly contain "
            "sensitive ancestors."
        )

    if risky_lineage != {1}:
        raise ValueError(
            "Risky trajectories do not contain "
            "sensitive ancestors."
        )

    print(
        "✓ Final current sensitivity is identical "
        "across benign and risky variants."
    )

    print(
        "✓ Sensitive ancestry differs between variants."
    )

    print(
        "✓ Hard-negative provenance condition verified."
    )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_jsonl(
        OUTPUT_FILE,
        generated_records,
    )

    summary_payload = {
        "dataset_name": (
            "SentinelAI Provenance Hard-Negative "
            "Experiment"
        ),
        "description": (
            "Controlled paired trajectories using "
            "real AMI and Enron source records. "
            "Benign and risky variants share the "
            "same tool/action sequence, while the "
            "risky variant contains a confidential "
            "ancestor in its provenance lineage."
        ),
        "controlled_metadata_warning": (
            "AMI and Enron source material is public. "
            "CONFIDENTIAL is experimental security "
            "metadata and must not be interpreted as "
            "the original source material being "
            "actually confidential."
        ),
        "target_trajectories_per_dataset": (
            TARGET_TRAJECTORIES_PER_DATASET
        ),
        "steps_per_trajectory": (
            STEPS_PER_TRAJECTORY
        ),
        **summary,
        "output_file": str(
            OUTPUT_FILE
        ),
    }

    with SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary_payload,
            file,
            indent=2,
        )

    print("\nSaved:")

    print(
        f"  ✓ {OUTPUT_FILE}"
    )

    print(
        f"  ✓ {SUMMARY_FILE}"
    )

    print("\n" + "=" * 70)
    print(
        "✓ STEP 57 COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)

    print(
        "\nThe new dataset contains paired benign/risky "
        "trajectories where current sensitivity is "
        "identical at the final step but provenance "
        "lineage differs."
    )


if __name__ == "__main__":
    main()