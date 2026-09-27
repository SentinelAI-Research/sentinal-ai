from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

AMI_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "ami"
    / "ami_security_experiment.jsonl"
)

ENRON_INPUT = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_experiment.jsonl"
)

AMI_FEATURES_WITHOUT_PROVENANCE = (
    ROOT
    / "data"
    / "processed"
    / "ami"
    / "ami_ml_features_without_provenance.jsonl"
)

AMI_FEATURES_WITH_PROVENANCE = (
    ROOT
    / "data"
    / "processed"
    / "ami"
    / "ami_ml_features_with_provenance.jsonl"
)

ENRON_FEATURES_WITHOUT_PROVENANCE = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_ml_features_without_provenance.jsonl"
)

ENRON_FEATURES_WITH_PROVENANCE = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_ml_features_with_provenance.jsonl"
)

OUTPUT_DIR = (
    ROOT
    / "data"
    / "processed"
    / "combined"
)

COMBINED_WITHOUT_PROVENANCE = (
    OUTPUT_DIR
    / "security_ml_features_without_provenance.jsonl"
)

COMBINED_WITH_PROVENANCE = (
    OUTPUT_DIR
    / "security_ml_features_with_provenance.jsonl"
)


EXPECTED_RECORDS_PER_DATASET = 600
EXPECTED_TRAJECTORIES_PER_DATASET = 100
EXPECTED_TOTAL_RECORDS = 1200
EXPECTED_TOTAL_TRAJECTORIES = 200
EXPECTED_STEPS = 6


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file does not exist: {path}"
        )

    records: list[dict] = []

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
                    f"Invalid JSON in {path} at "
                    f"line {line_number}."
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Record at {path}:{line_number} "
                    "is not a JSON object."
                )

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


def validate_experiment_dataset(
    records: list[dict],
    dataset_name: str,
) -> None:

    if len(records) != EXPECTED_RECORDS_PER_DATASET:
        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{EXPECTED_RECORDS_PER_DATASET} records, "
            f"found {len(records)}."
        )

    trajectory_ids = {
        record["trajectory_id"]
        for record in records
    }

    if (
        len(trajectory_ids)
        != EXPECTED_TRAJECTORIES_PER_DATASET
    ):
        raise RuntimeError(
            f"{dataset_name}: expected "
            f"{EXPECTED_TRAJECTORIES_PER_DATASET} "
            f"trajectories, found "
            f"{len(trajectory_ids)}."
        )

    counts = Counter(
        record["trajectory_id"]
        for record in records
    )

    for trajectory_id, count in counts.items():
        if count != EXPECTED_STEPS:
            raise RuntimeError(
                f"{dataset_name}: trajectory "
                f"{trajectory_id} contains "
                f"{count} records instead of "
                f"{EXPECTED_STEPS}."
            )

    pairs = [
        (
            record["trajectory_id"],
            record["step_number"],
        )
        for record in records
    ]

    if len(pairs) != len(set(pairs)):
        raise RuntimeError(
            f"{dataset_name}: duplicate "
            "trajectory/step pair detected."
        )

    expected_labels = {
        1: "SAFE",
        2: "EARLY_RISK",
        3: "EARLY_RISK",
        4: "EARLY_RISK",
        5: "EARLY_RISK",
        6: "VIOLATION",
    }

    for trajectory_id in trajectory_ids:

        trajectory = sorted(
            (
                record
                for record in records
                if record["trajectory_id"]
                == trajectory_id
            ),
            key=lambda record:
            record["step_number"],
        )

        steps = [
            record["step_number"]
            for record in trajectory
        ]

        if steps != list(range(1, EXPECTED_STEPS + 1)):
            raise RuntimeError(
                f"{dataset_name}: trajectory "
                f"{trajectory_id} has invalid "
                f"step sequence {steps}."
            )

        for record in trajectory:

            step = record["step_number"]

            if record["label"] != expected_labels[step]:
                raise RuntimeError(
                    f"{dataset_name}: trajectory "
                    f"{trajectory_id}, step {step}: "
                    f"unexpected label "
                    f"{record['label']}."
                )

    print(
        f"✓ {dataset_name}: "
        f"{len(records)} records, "
        f"{len(trajectory_ids)} trajectories"
    )


def add_dataset_source(
    records: list[dict],
    source_dataset: str,
) -> list[dict]:

    enriched_records = []

    for record in records:
        enriched = dict(record)

        enriched["source_dataset"] = source_dataset

        enriched_records.append(enriched)

    return enriched_records


def validate_feature_alignment(
    without_provenance: list[dict],
    with_provenance: list[dict],
) -> None:

    if len(without_provenance) != len(
        with_provenance
    ):
        raise RuntimeError(
            "Without-provenance and "
            "with-provenance datasets "
            "have different record counts."
        )

    identity_fields = {
        "trajectory_id",
        "scenario_id",
        "step_number",
        "total_steps",
        "trajectory_progress",
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
        "source_dataset",
    }

    for index, (
        base,
        provenance,
    ) in enumerate(
        zip(
            without_provenance,
            with_provenance,
        ),
        start=1,
    ):

        for field in identity_fields:

            if base.get(field) != provenance.get(field):
                raise RuntimeError(
                    f"Feature alignment mismatch "
                    f"at record {index}, "
                    f"field '{field}'."
                )

        forbidden_fields = {
            "provenance_depth",
            "transformation_depth",
            "has_provenance",
        }

        if forbidden_fields.intersection(
            base.keys()
        ):
            raise RuntimeError(
                "Without-provenance feature set "
                "contains provenance fields."
            )

        for field in forbidden_fields:

            if field not in provenance:
                raise RuntimeError(
                    f"Provenance feature set is "
                    f"missing '{field}'."
                )


def main() -> None:

    print("=" * 70)
    print("SENTINELAI — COMBINE SECURITY ML DATASETS")
    print("=" * 70)
    print()

    print("Loading experiment datasets...")

    ami_experiment = load_jsonl(
        AMI_INPUT
    )

    enron_experiment = load_jsonl(
        ENRON_INPUT
    )

    validate_experiment_dataset(
        ami_experiment,
        "AMI",
    )

    validate_experiment_dataset(
        enron_experiment,
        "Enron",
    )

    print()

    ami_without = load_jsonl(
        AMI_FEATURES_WITHOUT_PROVENANCE
    )

    ami_with = load_jsonl(
        AMI_FEATURES_WITH_PROVENANCE
    )

    enron_without = load_jsonl(
        ENRON_FEATURES_WITHOUT_PROVENANCE
    )

    enron_with = load_jsonl(
        ENRON_FEATURES_WITH_PROVENANCE
    )

    print(
        f"✓ AMI feature records loaded: "
        f"{len(ami_without)} / "
        f"{len(ami_with)}"
    )

    print(
        f"✓ Enron feature records loaded: "
        f"{len(enron_without)} / "
        f"{len(enron_with)}"
    )

    if len(ami_without) != EXPECTED_RECORDS_PER_DATASET:
        raise RuntimeError(
            "AMI without-provenance feature "
            "dataset does not contain 600 records."
        )

    if len(ami_with) != EXPECTED_RECORDS_PER_DATASET:
        raise RuntimeError(
            "AMI with-provenance feature "
            "dataset does not contain 600 records."
        )

    if len(enron_without) != EXPECTED_RECORDS_PER_DATASET:
        raise RuntimeError(
            "Enron without-provenance feature "
            "dataset does not contain 600 records."
        )

    if len(enron_with) != EXPECTED_RECORDS_PER_DATASET:
        raise RuntimeError(
            "Enron with-provenance feature "
            "dataset does not contain 600 records."
        )

    ami_without = add_dataset_source(
        ami_without,
        "ami",
    )

    ami_with = add_dataset_source(
        ami_with,
        "ami",
    )

    enron_without = add_dataset_source(
        enron_without,
        "enron",
    )

    enron_with = add_dataset_source(
        enron_with,
        "enron",
    )

    combined_without = (
        ami_without
        + enron_without
    )

    combined_with = (
        ami_with
        + enron_with
    )

    if len(combined_without) != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_RECORDS} "
            "combined without-provenance records, "
            f"found {len(combined_without)}."
        )

    if len(combined_with) != EXPECTED_TOTAL_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_TOTAL_RECORDS} "
            "combined with-provenance records, "
            f"found {len(combined_with)}."
        )

    validate_feature_alignment(
        combined_without,
        combined_with,
    )

    combined_trajectory_ids = {
        record["trajectory_id"]
        for record in combined_without
    }

    if (
        len(combined_trajectory_ids)
        != EXPECTED_TOTAL_TRAJECTORIES
    ):
        raise RuntimeError(
            f"Expected "
            f"{EXPECTED_TOTAL_TRAJECTORIES} "
            f"combined trajectories, found "
            f"{len(combined_trajectory_ids)}."
        )

    trajectory_pairs = [
        (
            record["trajectory_id"],
            record["step_number"],
        )
        for record in combined_without
    ]

    if len(trajectory_pairs) != len(
        set(trajectory_pairs)
    ):
        raise RuntimeError(
            "Duplicate trajectory/step pairs "
            "exist in the combined dataset."
        )

    source_counts = Counter(
        record["source_dataset"]
        for record in combined_without
    )

    expected_source_counts = {
        "ami": EXPECTED_RECORDS_PER_DATASET,
        "enron": EXPECTED_RECORDS_PER_DATASET,
    }

    if dict(source_counts) != expected_source_counts:
        raise RuntimeError(
            "Unexpected source distribution: "
            f"{dict(source_counts)}"
        )

    label_counts = Counter(
        record["label"]
        for record in combined_without
    )

    expected_label_counts = {
        "SAFE": 200,
        "EARLY_RISK": 800,
        "VIOLATION": 200,
    }

    if dict(label_counts) != expected_label_counts:
        raise RuntimeError(
            "Unexpected combined label distribution: "
            f"{dict(label_counts)}"
        )

    write_jsonl(
        COMBINED_WITHOUT_PROVENANCE,
        combined_without,
    )

    write_jsonl(
        COMBINED_WITH_PROVENANCE,
        combined_with,
    )

    print()
    print("✓ Combined record count = 1,200")
    print("✓ Combined trajectory count = 200")
    print("✓ AMI records = 600")
    print("✓ Enron records = 600")
    print(
        "✓ Source distribution:",
        dict(source_counts),
    )
    print(
        "✓ Label distribution:",
        dict(label_counts),
    )
    print(
        "✓ Without/with provenance feature "
        "alignment validated"
    )
    print(
        "✓ No duplicate trajectory/step pairs"
    )

    print()
    print("Output files:")
    print(
        f"  {COMBINED_WITHOUT_PROVENANCE}"
    )
    print(
        f"  {COMBINED_WITH_PROVENANCE}"
    )

    print()
    print("=" * 70)
    print("STEP 53 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()