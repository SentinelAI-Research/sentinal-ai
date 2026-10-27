"""Train both SentinelAI hard-negative models using split_v2.

Model A: current/action-level features, without provenance features.
Model B: current/action-level features plus provenance features.

Inputs:
    data/processed/hard_negative/split_v2/

Outputs:
    models/hard_negative/split_v2/

The original training script, models, and results are not overwritten.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = ROOT / "data" / "processed" / "hard_negative" / "split_v2"
MODEL_DIR = ROOT / "models" / "hard_negative" / "split_v2"

TRAIN_WITHOUT = SPLIT_DIR / "train_without_provenance.jsonl"
TEST_WITHOUT = SPLIT_DIR / "test_without_provenance.jsonl"
TRAIN_WITH = SPLIT_DIR / "train_with_provenance.jsonl"
TEST_WITH = SPLIT_DIR / "test_with_provenance.jsonl"

CLASSIFIER_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
}

PROVENANCE_FEATURES = {
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity_encoded",
}

EXCLUDED_FEATURES = {
    "trajectory_id",
    "scenario_id",
    "source_dataset",
    "step_number",
    "total_steps",
    "trajectory_progress",
    "label",
    "label_encoded",
    "tool_name",
    "transformation",
    "sensitivity",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")

    records = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}, line {line_number}") from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}, line {line_number}"
                )
            records.append(record)

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return records


def discover_numeric_features(records: list[dict]) -> list[str]:
    """Discover numeric input features, excluding labels and identifiers."""
    candidate_keys = set().union(*(record.keys() for record in records))
    feature_names = []

    for key in sorted(candidate_keys):
        if key in EXCLUDED_FEATURES:
            continue

        values = [record.get(key) for record in records]

        # A feature must be numeric wherever it is present.
        present_values = [value for value in values if value is not None]

        if not present_values:
            continue

        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in present_values
        ):
            feature_names.append(key)

    if not feature_names:
        raise ValueError("No numeric features discovered.")

    return feature_names


def validate_records(records: list[dict], name: str) -> None:
    for index, record in enumerate(records):
        required = ("trajectory_id", "step_number", "label_encoded")

        missing = [key for key in required if key not in record]
        if missing:
            raise ValueError(f"{name} row {index} is missing fields: {missing}")

        try:
            int(record["label_encoded"])
            int(record["step_number"])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{name} row {index} has a non-numeric label or step."
            ) from exc


def validate_alignment(
    records_a: list[dict],
    records_b: list[dict],
    partition_name: str,
) -> None:
    """Ensure both feature views contain the same examples in the same order."""
    keys_a = [
        (r["trajectory_id"], r["step_number"], r["label_encoded"]) for r in records_a
    ]
    keys_b = [
        (r["trajectory_id"], r["step_number"], r["label_encoded"]) for r in records_b
    ]

    if keys_a != keys_b:
        raise ValueError(
            f"{partition_name} datasets are not aligned by trajectory, "
            "step, and label."
        )

    print(f"{partition_name} feature-view alignment: PASS")


def build_matrix(
    records: list[dict],
    feature_names: list[str],
) -> list[list[float]]:
    matrix = []

    for index, record in enumerate(records):
        row = []

        for feature_name in feature_names:
            value = record.get(feature_name, 0)

            if value is None:
                value = 0

            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Non-numeric value for {feature_name!r} "
                    f"in record {index}: {value!r}"
                )

            row.append(float(value))

        matrix.append(row)

    return matrix


def extract_labels(records: list[dict]) -> list[int]:
    return [int(record["label_encoded"]) for record in records]


def variant_of(record: dict) -> str:
    variant = record.get("variant")
    if variant:
        return str(variant)

    trajectory_id = str(record.get("trajectory_id", "")).lower()
    if trajectory_id.endswith("-provenance_risky"):
        return "provenance_risky"
    if trajectory_id.endswith("-benign"):
        return "benign"
    return "unknown"


def train_model(
    train_records: list[dict],
    test_records: list[dict],
    model_name: str,
    use_provenance: bool,
) -> dict:
    feature_names = discover_numeric_features(train_records)

    if not use_provenance:
        feature_names = [
            feature for feature in feature_names if feature not in PROVENANCE_FEATURES
        ]

    if not feature_names:
        raise ValueError(f"No usable features for {model_name}.")

    X_train = build_matrix(train_records, feature_names)
    y_train = extract_labels(train_records)
    X_test = build_matrix(test_records, feature_names)
    y_test = extract_labels(test_records)

    model = HistGradientBoostingClassifier(**CLASSIFIER_CONFIG)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / f"{model_name}.joblib"
    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"

    # Never overwrite a previous v2 model accidentally.
    if model_path.exists() or metadata_path.exists():
        raise FileExistsError(
            f"Output already exists for {model_name} in {MODEL_DIR}. "
            "Inspect existing outputs before rerunning."
        )

    metadata = {
        "model_name": model_name,
        "experiment": "provenance_hard_negative_split_v2",
        "use_provenance": use_provenance,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_config": CLASSIFIER_CONFIG,
        "feature_names": feature_names,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_trajectories": len(
            {record["trajectory_id"] for record in train_records}
        ),
        "test_trajectories": len({record["trajectory_id"] for record in test_records}),
        "train_variant_counts": dict(
            Counter(variant_of(record) for record in train_records)
        ),
        "test_variant_counts": dict(
            Counter(variant_of(record) for record in test_records)
        ),
        "train_label_counts": dict(
            Counter(str(record["label_encoded"]) for record in train_records)
        ),
        "test_label_counts": dict(
            Counter(str(record["label_encoded"]) for record in test_records)
        ),
        "classification_report": report,
    }

    joblib.dump(
        {
            "model": model,
            "feature_names": feature_names,
            "metadata": metadata,
        },
        model_path,
    )

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print(model_name)
    print("=" * 72)
    print(f"Provenance enabled: {use_provenance}")
    print(f"Train rows:         {len(train_records)}")
    print(f"Test rows:          {len(test_records)}")
    print(f"Train trajectories: {metadata['train_trajectories']}")
    print(f"Test trajectories:  {metadata['test_trajectories']}")
    print(f"Features ({len(feature_names)}):")

    for feature in feature_names:
        print(f"  - {feature}")

    print(f"\nAccuracy: {report['accuracy']:.4f}")
    print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Metadata saved: {metadata_path}")

    return metadata


def main() -> None:
    print("=" * 72)
    print("SENTINELAI — HARD-NEGATIVE MODEL TRAINING, SPLIT V2")
    print("=" * 72)

    train_without = load_jsonl(TRAIN_WITHOUT)
    test_without = load_jsonl(TEST_WITHOUT)
    train_with = load_jsonl(TRAIN_WITH)
    test_with = load_jsonl(TEST_WITH)

    for name, records in (
        ("train_without", train_without),
        ("test_without", test_without),
        ("train_with", train_with),
        ("test_with", test_with),
    ):
        validate_records(records, name)

    validate_alignment(train_without, train_with, "Training")
    validate_alignment(test_without, test_with, "Testing")

    # Independently verify that the train and test trajectory sets do not
    # overlap. The split-generation script also groups paired source variants.
    train_ids = {r["trajectory_id"] for r in train_without}
    test_ids = {r["trajectory_id"] for r in test_without}

    if train_ids & test_ids:
        raise ValueError("Train/test trajectory overlap detected.")

    print("Train/test trajectory overlap: NONE")
    print(f"Training label IDs: {sorted(set(extract_labels(train_without)))}")
    print(f"Test label IDs:     {sorted(set(extract_labels(test_without)))}")

    print("\nTraining non-provenance model...")
    without_metadata = train_model(
        train_records=train_without,
        test_records=test_without,
        model_name="security_risk_hard_negative_without_provenance_v2",
        use_provenance=False,
    )

    print("\nTraining provenance-aware model...")
    with_metadata = train_model(
        train_records=train_with,
        test_records=test_with,
        model_name="security_risk_hard_negative_with_provenance_v2",
        use_provenance=True,
    )

    comparison = {
        "experiment": "provenance_hard_negative_split_v2",
        "same_split": True,
        "without_provenance": {
            "accuracy": without_metadata["classification_report"]["accuracy"],
            "macro_f1": without_metadata["classification_report"]["macro avg"][
                "f1-score"
            ],
        },
        "with_provenance": {
            "accuracy": with_metadata["classification_report"]["accuracy"],
            "macro_f1": with_metadata["classification_report"]["macro avg"]["f1-score"],
        },
    }

    comparison_path = MODEL_DIR / "model_comparison_v2.json"
    comparison_path.write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("SPLIT V2 TRAINING COMPLETE")
    print("=" * 72)
    print(json.dumps(comparison, indent=2))
    print(f"\nComparison saved: {comparison_path}")
    print("Original models and result files were not overwritten.")


if __name__ == "__main__":
    main()