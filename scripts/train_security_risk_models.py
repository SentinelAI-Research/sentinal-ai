"""
SENTINELAI — Step 55
Train provenance-aware and non-provenance ML risk models.

Two identical HistGradientBoostingClassifier models are trained:

1. WITHOUT provenance
2. WITH provenance

The models use the same train/test trajectory split created in Step 54.

Important:
- Label columns are never used as features.
- trajectory_id and scenario_id are never used as features.
- step_number and trajectory_progress are intentionally excluded because
  the current controlled dataset has labels aligned with trajectory stages.
- Both models use the same classifier configuration.
- The only experimental difference is provenance information.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path("data/processed/combined")
MODEL_DIR = Path("models")

TRAIN_WITHOUT = BASE_DIR / "train_without_provenance.jsonl"
TEST_WITHOUT = BASE_DIR / "test_without_provenance.jsonl"

TRAIN_WITH = BASE_DIR / "train_with_provenance.jsonl"
TEST_WITH = BASE_DIR / "test_with_provenance.jsonl"

MODEL_WITHOUT_PATH = MODEL_DIR / "security_risk_without_provenance.joblib"
MODEL_WITH_PATH = MODEL_DIR / "security_risk_with_provenance.joblib"

FEATURES_WITHOUT_PATH = (
    MODEL_DIR / "security_risk_features_without_provenance.json"
)

FEATURES_WITH_PATH = (
    MODEL_DIR / "security_risk_features_with_provenance.json"
)

METADATA_WITHOUT_PATH = (
    MODEL_DIR / "security_risk_metadata_without_provenance.json"
)

METADATA_WITH_PATH = (
    MODEL_DIR / "security_risk_metadata_with_provenance.json"
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RANDOM_SEED = 42

CLASSIFIER_CONFIG = {
    "learning_rate": 0.05,
    "max_iter": 200,
    "max_leaf_nodes": 15,
    "l2_regularization": 1.0,
    "random_state": RANDOM_SEED,
}


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# These are intentionally excluded because they identify the trajectory,
# scenario, source record, or controlled trajectory position.
EXCLUDED_FEATURES = {
    "trajectory_id",
    "scenario_id",
    "source_dataset",
    "step_number",
    "total_steps",
    "trajectory_progress",
    "label",
    "label_encoded",
    "record_id",
    "object_id",
    "source_object_id",
    "content_hash",
}


# Provenance-specific features are only allowed in the provenance-aware
# model.
PROVENANCE_FEATURES = {
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
}


# ---------------------------------------------------------------------------
# JSONL utilities
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required dataset file does not exist: {path}"
        )

    records: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} at line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object in {path} at line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


# ---------------------------------------------------------------------------
# Feature discovery
# ---------------------------------------------------------------------------

def discover_numeric_features(
    records: list[dict[str, Any]],
    include_provenance: bool,
) -> list[str]:
    """
    Discover numeric features that are valid for ML.

    We deliberately avoid blindly converting every JSON field into a model
    feature. Only numeric fields are considered, and explicitly excluded
    fields are removed.
    """

    candidate_names: set[str] = set()

    for record in records:
        for key, value in record.items():

            if key in EXCLUDED_FEATURES:
                continue

            if not include_provenance and key in PROVENANCE_FEATURES:
                continue

            if isinstance(value, bool):
                candidate_names.add(key)

            elif isinstance(value, (int, float)) and not isinstance(
                value, bool
            ):
                candidate_names.add(key)

    # Sort so the feature order is deterministic and reproducible.
    return sorted(candidate_names)


def validate_feature_values(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> None:
    for record_index, record in enumerate(records):

        for feature in feature_names:

            if feature not in record:
                raise ValueError(
                    f"Missing feature '{feature}' in record "
                    f"{record_index}."
                )

            value = record[feature]

            if isinstance(value, bool):
                continue

            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"Feature '{feature}' must be numeric. "
                    f"Found {type(value).__name__}."
                )

            if not np.isfinite(float(value)):
                raise ValueError(
                    f"Non-finite value found for feature '{feature}' "
                    f"in record {record_index}."
                )


# ---------------------------------------------------------------------------
# Feature matrix construction
# ---------------------------------------------------------------------------

def build_feature_matrix(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:

    matrix = np.array(
        [
            [
                float(record[feature])
                for feature in feature_names
            ]
            for record in records
        ],
        dtype=float,
    )

    return matrix


def build_labels(
    records: list[dict[str, Any]],
) -> np.ndarray:

    if not all("label_encoded" in record for record in records):
        raise ValueError(
            "Every record must contain 'label_encoded'."
        )

    labels = np.array(
        [
            int(record["label_encoded"])
            for record in records
        ],
        dtype=int,
    )

    return labels


# ---------------------------------------------------------------------------
# Dataset validation
# ---------------------------------------------------------------------------

def validate_train_test_alignment(
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:

    if not train_records:
        raise ValueError("Training dataset is empty.")

    if not test_records:
        raise ValueError("Test dataset is empty.")

    train_trajectories = {
        str(record["trajectory_id"])
        for record in train_records
    }

    test_trajectories = {
        str(record["trajectory_id"])
        for record in test_records
    }

    overlap = train_trajectories & test_trajectories

    if overlap:
        raise ValueError(
            "TRAJECTORY LEAKAGE DETECTED.\n"
            f"Overlapping trajectories: {sorted(overlap)}"
        )

    print(
        f"✓ Train trajectories: {len(train_trajectories)}"
    )

    print(
        f"✓ Test trajectories:  {len(test_trajectories)}"
    )

    print("✓ No train/test trajectory overlap.")


def validate_label_distribution(
    name: str,
    labels: np.ndarray,
) -> None:

    unique, counts = np.unique(
        labels,
        return_counts=True,
    )

    print(f"\n{name} label distribution:")

    for label, count in zip(unique, counts):
        print(f"  Class {label}: {count}")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> HistGradientBoostingClassifier:

    model = HistGradientBoostingClassifier(
        **CLASSIFIER_CONFIG
    )

    model.fit(
        X_train,
        y_train,
    )

    return model


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------

def save_json(
    path: Path,
    payload: dict[str, Any],
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open("w", encoding="utf-8") as file:
        json.dump(
            payload,
            file,
            indent=2,
        )


def save_model_artifacts(
    model: HistGradientBoostingClassifier,
    model_path: Path,
    feature_names: list[str],
    feature_path: Path,
    metadata_path: Path,
    include_provenance: bool,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
) -> None:

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        model_path,
    )

    save_json(
        feature_path,
        {
            "model_type": "HistGradientBoostingClassifier",
            "include_provenance": include_provenance,
            "features": feature_names,
        },
    )

    train_trajectories = {
        str(record["trajectory_id"])
        for record in train_records
    }

    test_trajectories = {
        str(record["trajectory_id"])
        for record in test_records
    }

    save_json(
        metadata_path,
        {
            "model_type": "HistGradientBoostingClassifier",
            "random_seed": RANDOM_SEED,
            "classifier_config": CLASSIFIER_CONFIG,
            "include_provenance": include_provenance,
            "feature_count": len(feature_names),
            "features": feature_names,
            "train_records": len(train_records),
            "test_records": len(test_records),
            "train_trajectories": len(train_trajectories),
            "test_trajectories": len(test_trajectories),
            "excluded_features": sorted(EXCLUDED_FEATURES),
        },
    )


# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(
    name: str,
    train_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    include_provenance: bool,
    model_path: Path,
    feature_path: Path,
    metadata_path: Path,
) -> None:

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    feature_names = discover_numeric_features(
        train_records,
        include_provenance=include_provenance,
    )

    if not feature_names:
        raise ValueError(
            f"No valid numeric features discovered for {name}."
        )

    validate_feature_values(
        train_records,
        feature_names,
    )

    validate_feature_values(
        test_records,
        feature_names,
    )

    X_train = build_feature_matrix(
        train_records,
        feature_names,
    )

    y_train = build_labels(
        train_records,
    )

    X_test = build_feature_matrix(
        test_records,
        feature_names,
    )

    y_test = build_labels(
        test_records,
    )

    print(f"\nNumber of features: {len(feature_names)}")

    print("\nFeatures:")
    for feature in feature_names:
        print(f"  - {feature}")

    print(
        f"\nTraining matrix: {X_train.shape}"
    )

    print(
        f"Testing matrix:  {X_test.shape}"
    )

    validate_label_distribution(
        "Training",
        y_train,
    )

    validate_label_distribution(
        "Testing",
        y_test,
    )

    print("\nTraining HistGradientBoostingClassifier...")

    model = train_model(
        X_train,
        y_train,
    )

    # Basic prediction sanity check.
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    print("\nPrediction sanity check:")
    print(
        f"  Training predictions: {len(train_predictions)}"
    )
    print(
        f"  Testing predictions:  {len(test_predictions)}"
    )

    save_model_artifacts(
        model=model,
        model_path=model_path,
        feature_names=feature_names,
        feature_path=feature_path,
        metadata_path=metadata_path,
        include_provenance=include_provenance,
        train_records=train_records,
        test_records=test_records,
    )

    print("\nSaved artifacts:")
    print(f"  ✓ {model_path}")
    print(f"  ✓ {feature_path}")
    print(f"  ✓ {metadata_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("SENTINELAI — STEP 55")
    print("ML RISK MODEL TRAINING")
    print("=" * 70)

    print("\nLoading datasets...")

    train_without = load_jsonl(
        TRAIN_WITHOUT
    )

    test_without = load_jsonl(
        TEST_WITHOUT
    )

    train_with = load_jsonl(
        TRAIN_WITH
    )

    test_with = load_jsonl(
        TEST_WITH
    )

    print(
        f"Without-provenance train records: "
        f"{len(train_without)}"
    )

    print(
        f"Without-provenance test records:  "
        f"{len(test_without)}"
    )

    print(
        f"With-provenance train records:    "
        f"{len(train_with)}"
    )

    print(
        f"With-provenance test records:     "
        f"{len(test_with)}"
    )

    # Confirm that both feature views have identical trajectory partitions.
    validate_train_test_alignment(
        train_without,
        test_without,
    )

    validate_train_test_alignment(
        train_with,
        test_with,
    )

    without_train_ids = {
        str(record["trajectory_id"])
        for record in train_without
    }

    with_train_ids = {
        str(record["trajectory_id"])
        for record in train_with
    }

    without_test_ids = {
        str(record["trajectory_id"])
        for record in test_without
    }

    with_test_ids = {
        str(record["trajectory_id"])
        for record in test_with
    }

    if without_train_ids != with_train_ids:
        raise ValueError(
            "Train trajectory partitions differ between "
            "provenance views."
        )

    if without_test_ids != with_test_ids:
        raise ValueError(
            "Test trajectory partitions differ between "
            "provenance views."
        )

    print(
        "✓ Both models use the exact same trajectory split."
    )

    # ------------------------------------------------------------------
    # Model A — WITHOUT PROVENANCE
    # ------------------------------------------------------------------

    run_experiment(
        name="MODEL A — WITHOUT PROVENANCE",
        train_records=train_without,
        test_records=test_without,
        include_provenance=False,
        model_path=MODEL_WITHOUT_PATH,
        feature_path=FEATURES_WITHOUT_PATH,
        metadata_path=METADATA_WITHOUT_PATH,
    )

    # ------------------------------------------------------------------
    # Model B — WITH PROVENANCE
    # ------------------------------------------------------------------

    run_experiment(
        name="MODEL B — WITH PROVENANCE",
        train_records=train_with,
        test_records=test_with,
        include_provenance=True,
        model_path=MODEL_WITH_PATH,
        feature_path=FEATURES_WITH_PATH,
        metadata_path=METADATA_WITH_PATH,
    )

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------

    print("\n" + "=" * 70)
    print("✓ STEP 55 COMPLETED")
    print("=" * 70)

    print("\nModels trained:")
    print(
        "  1. security_risk_without_provenance.joblib"
    )
    print(
        "  2. security_risk_with_provenance.joblib"
    )

    print(
        "\nThe models have been trained and saved."
    )

    print(
        "\nEvaluation is intentionally NOT performed here."
    )

    print(
        "Step 56 will evaluate both models on the held-out "
        "test trajectories."
    )


if __name__ == "__main__":
    main()