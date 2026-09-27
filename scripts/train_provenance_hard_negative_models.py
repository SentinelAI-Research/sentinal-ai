"""
Train provenance-aware and non-provenance ML risk models
on the provenance hard-negative dataset.

Model A:
    Uses current/action-level features only.

Model B:
    Uses current/action-level features + provenance features.

The same train/test trajectory split is used for both models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "hard_negative"
MODEL_DIR = ROOT / "models" / "hard_negative"
RESULT_DIR = ROOT / "results" / "ml_evaluation"

TRAIN_WITHOUT = DATA_DIR / "train_without_provenance.jsonl"
TEST_WITHOUT = DATA_DIR / "test_without_provenance.jsonl"

TRAIN_WITH = DATA_DIR / "train_with_provenance.jsonl"
TEST_WITH = DATA_DIR / "test_with_provenance.jsonl"


CLASSIFIER_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": 42,
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
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}"
                ) from exc

    if not records:
        raise ValueError(f"Dataset is empty: {path}")

    return records


def discover_numeric_features(records: list[dict]) -> list[str]:
    feature_names = []

    for key in records[0].keys():
        if key in EXCLUDED_FEATURES:
            continue

        values = [record.get(key) for record in records[:100]]

        if all(
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            for value in values
            if value is not None
        ):
            feature_names.append(key)

    if not feature_names:
        raise ValueError("No numeric features discovered.")

    return sorted(feature_names)


def build_matrix(
    records: list[dict],
    feature_names: list[str],
) -> list[list[float]]:
    matrix = []

    for record in records:
        row = []

        for feature_name in feature_names:
            value = record.get(feature_name, 0)

            if value is None:
                value = 0

            row.append(float(value))

        matrix.append(row)

    return matrix


def extract_labels(records: list[dict]) -> list[int]:
    return [int(record["label_encoded"]) for record in records]


def train_model(
    train_records: list[dict],
    test_records: list[dict],
    model_name: str,
    use_provenance: bool,
) -> dict:

    feature_names = discover_numeric_features(train_records)

    if not use_provenance:
        feature_names = [
            feature
            for feature in feature_names
            if feature
            not in {
                "provenance_depth",
                "transformation_depth",
                "has_provenance",
                "has_sensitive_ancestor",
                "sensitive_ancestor_count",
                "max_ancestor_sensitivity_encoded",
            }
        ]

    if not feature_names:
        raise ValueError(
            f"No usable features found for {model_name}."
        )

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

    metadata = {
        "model_name": model_name,
        "use_provenance": use_provenance,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_config": CLASSIFIER_CONFIG,
        "feature_names": feature_names,
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_trajectories": len(
            {record["trajectory_id"] for record in train_records}
        ),
        "test_trajectories": len(
            {record["trajectory_id"] for record in test_records}
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

    metadata_path = MODEL_DIR / f"{model_name}_metadata.json"

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print(f"{model_name}")
    print("=" * 70)
    print(f"Provenance enabled : {use_provenance}")
    print(f"Train records      : {len(train_records)}")
    print(f"Test records       : {len(test_records)}")
    print(f"Train trajectories : {metadata['train_trajectories']}")
    print(f"Test trajectories  : {metadata['test_trajectories']}")
    print(f"Features ({len(feature_names)}):")

    for feature in feature_names:
        print(f"  - {feature}")

    print()
    print(
        f"Accuracy: "
        f"{report['accuracy']:.4f}"
    )

    print(
        f"Macro F1: "
        f"{report['macro avg']['f1-score']:.4f}"
    )

    print()
    print(f"Model saved to: {model_path}")
    print(f"Metadata saved to: {metadata_path}")
    print()


def main() -> None:
    print("=" * 70)
    print("SENTINELAI — PROVENANCE HARD-NEGATIVE MODEL TRAINING")
    print("=" * 70)

    train_without = load_jsonl(TRAIN_WITHOUT)
    test_without = load_jsonl(TEST_WITHOUT)

    train_with = load_jsonl(TRAIN_WITH)
    test_with = load_jsonl(TEST_WITH)

    if len(train_without) != len(train_with):
        raise ValueError(
            "Train datasets for the two views are not aligned."
        )

    if len(test_without) != len(test_with):
        raise ValueError(
            "Test datasets for the two views are not aligned."
        )

    train_without_ids = [
        (record["trajectory_id"], record["step_number"])
        for record in train_without
    ]

    train_with_ids = [
        (record["trajectory_id"], record["step_number"])
        for record in train_with
    ]

    test_without_ids = [
        (record["trajectory_id"], record["step_number"])
        for record in test_without
    ]

    test_with_ids = [
        (record["trajectory_id"], record["step_number"])
        for record in test_with
    ]

    if train_without_ids != train_with_ids:
        raise ValueError(
            "Train trajectory/step alignment differs between views."
        )

    if test_without_ids != test_with_ids:
        raise ValueError(
            "Test trajectory/step alignment differs between views."
        )

    print()
    print("Dataset alignment verified.")
    print()

    train_model(
        train_records=train_without,
        test_records=test_without,
        model_name="security_risk_hard_negative_without_provenance",
        use_provenance=False,
    )

    train_model(
        train_records=train_with,
        test_records=test_with,
        model_name="security_risk_hard_negative_with_provenance",
        use_provenance=True,
    )

    print("=" * 70)
    print("STEP 60 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()