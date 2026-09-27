"""
SENTINELAI — Step 56
Evaluate provenance-aware and non-provenance ML risk models.

This script evaluates the two models trained in Step 55 on the same
held-out test trajectories:

1. WITHOUT PROVENANCE
2. WITH PROVENANCE

Metrics:
- Accuracy
- Precision
- Recall
- F1
- Confusion matrix
- Per-class precision/recall/F1
- AUPRC (one-vs-rest macro)
- False-positive rate for risk detection

Important:
- The test set is never used for training.
- Both models are evaluated on the exact same trajectories.
- Feature lists saved during Step 55 are reused.
- No feature engineering is performed differently between models.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_DIR = Path("data/processed/combined")
MODEL_DIR = Path("models")
RESULTS_DIR = Path("results/ml_evaluation")

TEST_WITHOUT = BASE_DIR / "test_without_provenance.jsonl"
TEST_WITH = BASE_DIR / "test_with_provenance.jsonl"

MODEL_WITHOUT = MODEL_DIR / "security_risk_without_provenance.joblib"
MODEL_WITH = MODEL_DIR / "security_risk_with_provenance.joblib"

FEATURES_WITHOUT = (
    MODEL_DIR / "security_risk_features_without_provenance.json"
)

FEATURES_WITH = (
    MODEL_DIR / "security_risk_features_with_provenance.json"
)

RESULTS_JSON = RESULTS_DIR / "security_risk_evaluation.json"

RESULTS_TEXT = RESULTS_DIR / "security_risk_evaluation.txt"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_NAMES = {
    0: "SAFE",
    1: "EARLY_RISK",
    2: "VIOLATION",
}

LABEL_ORDER = [0, 1, 2]


# ---------------------------------------------------------------------------
# JSONL loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}"
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
                    f"Expected JSON object in {path} "
                    f"at line {line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(
            f"No records found in {path}"
        )

    return records


# ---------------------------------------------------------------------------
# Model metadata
# ---------------------------------------------------------------------------

def load_feature_list(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(
            f"Feature metadata file not found: {path}"
        )

    with path.open("r", encoding="utf-8") as file:
        metadata = json.load(file)

    features = metadata.get("features")

    if not isinstance(features, list) or not features:
        raise ValueError(
            f"No feature list found in {path}"
        )

    return [str(feature) for feature in features]


# ---------------------------------------------------------------------------
# Feature matrix
# ---------------------------------------------------------------------------

def build_feature_matrix(
    records: list[dict[str, Any]],
    feature_names: list[str],
) -> np.ndarray:

    rows: list[list[float]] = []

    for record_index, record in enumerate(records):

        row: list[float] = []

        for feature in feature_names:

            if feature not in record:
                raise ValueError(
                    f"Feature '{feature}' missing from test record "
                    f"{record_index}."
                )

            value = record[feature]

            if isinstance(value, bool):
                row.append(float(value))

            elif isinstance(value, (int, float)):
                numeric_value = float(value)

                if not np.isfinite(numeric_value):
                    raise ValueError(
                        f"Non-finite value for feature '{feature}' "
                        f"in test record {record_index}."
                    )

                row.append(numeric_value)

            else:
                raise TypeError(
                    f"Feature '{feature}' must be numeric. "
                    f"Found {type(value).__name__}."
                )

        rows.append(row)

    return np.asarray(
        rows,
        dtype=float,
    )


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def build_labels(
    records: list[dict[str, Any]],
) -> np.ndarray:

    labels: list[int] = []

    for record_index, record in enumerate(records):

        if "label_encoded" not in record:
            raise ValueError(
                f"Missing label_encoded in test record "
                f"{record_index}."
            )

        labels.append(
            int(record["label_encoded"])
        )

    return np.asarray(
        labels,
        dtype=int,
    )


# ---------------------------------------------------------------------------
# Trajectory validation
# ---------------------------------------------------------------------------

def validate_test_sets(
    test_without: list[dict[str, Any]],
    test_with: list[dict[str, Any]],
) -> None:

    without_ids = {
        str(record["trajectory_id"])
        for record in test_without
    }

    with_ids = {
        str(record["trajectory_id"])
        for record in test_with
    }

    if without_ids != with_ids:
        raise ValueError(
            "The two test datasets do not contain the same "
            "trajectory IDs."
        )

    print(
        f"✓ Test trajectories: {len(without_ids)}"
    )

    print(
        "✓ Both models are evaluated on the same trajectories."
    )


# ---------------------------------------------------------------------------
# AUPRC
# ---------------------------------------------------------------------------

def calculate_macro_auprc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> float:

    """
    Calculate macro one-vs-rest Average Precision.

    Each class is converted to a binary target:
        class == current_class

    Average precision is then calculated for each class and averaged.
    """

    scores: list[float] = []

    for class_index in LABEL_ORDER:

        binary_true = (
            y_true == class_index
        ).astype(int)

        class_probabilities = probabilities[:, class_index]

        score = average_precision_score(
            binary_true,
            class_probabilities,
        )

        scores.append(float(score))

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# False-positive rate
# ---------------------------------------------------------------------------

def calculate_risk_false_positive_rate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:

    """
    Calculate FPR for the binary question:

        SAFE vs RISK

    EARLY_RISK and VIOLATION are treated as RISK.

    A false positive is:
        actual SAFE
        predicted RISK
    """

    actual_safe = (
        y_true == 0
    )

    predicted_risk = (
        y_pred != 0
    )

    false_positives = np.sum(
        actual_safe & predicted_risk
    )

    actual_negatives = np.sum(
        actual_safe
    )

    if actual_negatives == 0:
        return float("nan")

    return float(
        false_positives / actual_negatives
    )


# ---------------------------------------------------------------------------
# Single model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model_name: str,
    model_path: Path,
    feature_path: Path,
    test_records: list[dict[str, Any]],
) -> dict[str, Any]:

    print("\n" + "=" * 70)
    print(model_name)
    print("=" * 70)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    feature_names = load_feature_list(
        feature_path
    )

    print(
        f"\nLoaded model: {model_path}"
    )

    print(
        f"Number of features: {len(feature_names)}"
    )

    print("\nFeatures used:")

    for feature in feature_names:
        print(f"  - {feature}")

    X_test = build_feature_matrix(
        test_records,
        feature_names,
    )

    y_test = build_labels(
        test_records,
    )

    print(
        f"\nTest matrix shape: {X_test.shape}"
    )

    print(
        f"Test labels:       {y_test.shape}"
    )

    # ---------------------------------------------------------------
    # Predictions
    # ---------------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    probabilities = model.predict_proba(
        X_test
    )

    # ---------------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        y_pred,
    )

    precision_macro = precision_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    recall_macro = recall_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    f1_macro = f1_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="macro",
        zero_division=0,
    )

    precision_weighted = precision_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    recall_weighted = recall_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    f1_weighted = f1_score(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        average="weighted",
        zero_division=0,
    )

    macro_auprc = calculate_macro_auprc(
        y_test,
        probabilities,
    )

    risk_fpr = calculate_risk_false_positive_rate(
        y_test,
        y_pred,
    )

    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
    )

    report = classification_report(
        y_test,
        y_pred,
        labels=LABEL_ORDER,
        target_names=[
            LABEL_NAMES[index]
            for index in LABEL_ORDER
        ],
        output_dict=True,
        zero_division=0,
    )

    # ---------------------------------------------------------------
    # Print metrics
    # ---------------------------------------------------------------

    print("\nOverall metrics:")
    print(
        f"  Accuracy:          {accuracy:.4f}"
    )
    print(
        f"  Macro Precision:   {precision_macro:.4f}"
    )
    print(
        f"  Macro Recall:      {recall_macro:.4f}"
    )
    print(
        f"  Macro F1:          {f1_macro:.4f}"
    )
    print(
        f"  Weighted Precision:{precision_weighted:.4f}"
    )
    print(
        f"  Weighted Recall:   {recall_weighted:.4f}"
    )
    print(
        f"  Weighted F1:       {f1_weighted:.4f}"
    )
    print(
        f"  Macro AUPRC:       {macro_auprc:.4f}"
    )

    if np.isnan(risk_fpr):
        print(
            "  Risk FPR:          N/A"
        )
    else:
        print(
            f"  Risk FPR:          {risk_fpr:.4f}"
        )

    # ---------------------------------------------------------------
    # Confusion matrix
    # ---------------------------------------------------------------

    print("\nConfusion matrix:")
    print(
        "Rows = actual | Columns = predicted"
    )

    print(
        "                 SAFE  EARLY_RISK  VIOLATION"
    )

    for row_index, row in enumerate(matrix):

        print(
            f"{LABEL_NAMES[row_index]:<16}"
            f"{row[0]:>5}"
            f"{row[1]:>12}"
            f"{row[2]:>11}"
        )

    # ---------------------------------------------------------------
    # Per-class metrics
    # ---------------------------------------------------------------

    print("\nPer-class metrics:")

    for class_index in LABEL_ORDER:

        class_name = LABEL_NAMES[class_index]

        class_metrics = report[class_name]

        print(
            f"\n  {class_name}"
        )

        print(
            f"    Precision: {class_metrics['precision']:.4f}"
        )

        print(
            f"    Recall:    {class_metrics['recall']:.4f}"
        )

        print(
            f"    F1:        {class_metrics['f1-score']:.4f}"
        )

        print(
            f"    Support:   {int(class_metrics['support'])}"
        )

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "feature_path": str(feature_path),
        "feature_count": len(feature_names),
        "features": feature_names,
        "test_records": len(test_records),
        "test_trajectories": len(
            {
                str(record["trajectory_id"])
                for record in test_records
            }
        ),
        "metrics": {
            "accuracy": float(accuracy),
            "macro_precision": float(precision_macro),
            "macro_recall": float(recall_macro),
            "macro_f1": float(f1_macro),
            "weighted_precision": float(precision_weighted),
            "weighted_recall": float(recall_weighted),
            "weighted_f1": float(f1_weighted),
            "macro_auprc": float(macro_auprc),
            "risk_false_positive_rate": (
                None
                if np.isnan(risk_fpr)
                else float(risk_fpr)
            ),
        },
        "confusion_matrix": matrix.tolist(),
        "classification_report": report,
    }


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def build_comparison(
    without_results: dict[str, Any],
    with_results: dict[str, Any],
) -> dict[str, Any]:

    without_metrics = without_results["metrics"]
    with_metrics = with_results["metrics"]

    comparison: dict[str, Any] = {}

    for metric_name in without_metrics:

        without_value = without_metrics[metric_name]
        with_value = with_metrics[metric_name]

        if (
            without_value is None
            or with_value is None
        ):
            comparison[metric_name] = None
        else:
            comparison[metric_name] = (
                float(with_value) - float(without_value)
            )

    return comparison


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    results: dict[str, Any],
) -> None:

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_JSON.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            results,
            file,
            indent=2,
        )

    # Human-readable summary.
    with RESULTS_TEXT.open(
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "SENTINELAI — ML MODEL EVALUATION\n"
        )

        file.write(
            "=" * 70 + "\n\n"
        )

        for model_key in [
            "without_provenance",
            "with_provenance",
        ]:

            model_results = results[model_key]

            file.write(
                f"{model_results['model_name']}\n"
            )

            file.write(
                "-" * 70 + "\n"
            )

            for metric, value in model_results[
                "metrics"
            ].items():

                if value is None:
                    file.write(
                        f"{metric}: N/A\n"
                    )
                else:
                    file.write(
                        f"{metric}: {value:.6f}\n"
                    )

            file.write("\n")

        file.write(
            "WITH PROVENANCE MINUS WITHOUT PROVENANCE\n"
        )

        file.write(
            "-" * 70 + "\n"
        )

        for metric, value in results[
            "comparison"
        ].items():

            if value is None:
                file.write(
                    f"{metric}: N/A\n"
                )
            else:
                file.write(
                    f"{metric}: {value:+.6f}\n"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    print("=" * 70)
    print("SENTINELAI — STEP 56")
    print("ML MODEL EVALUATION")
    print("=" * 70)

    print("\nLoading held-out test datasets...")

    test_without = load_jsonl(
        TEST_WITHOUT
    )

    test_with = load_jsonl(
        TEST_WITH
    )

    print(
        f"Without-provenance test records: "
        f"{len(test_without)}"
    )

    print(
        f"With-provenance test records:    "
        f"{len(test_with)}"
    )

    validate_test_sets(
        test_without,
        test_with,
    )

    # ---------------------------------------------------------------
    # Evaluate Model A
    # ---------------------------------------------------------------

    without_results = evaluate_model(
        model_name="MODEL A — WITHOUT PROVENANCE",
        model_path=MODEL_WITHOUT,
        feature_path=FEATURES_WITHOUT,
        test_records=test_without,
    )

    # ---------------------------------------------------------------
    # Evaluate Model B
    # ---------------------------------------------------------------

    with_results = evaluate_model(
        model_name="MODEL B — WITH PROVENANCE",
        model_path=MODEL_WITH,
        feature_path=FEATURES_WITH,
        test_records=test_with,
    )

    # ---------------------------------------------------------------
    # Comparison
    # ---------------------------------------------------------------

    comparison = build_comparison(
        without_results,
        with_results,
    )

    print("\n" + "=" * 70)
    print("PROVENANCE COMPARISON")
    print("=" * 70)

    print(
        "\nMetric                 Without      With       Difference"
    )

    print(
        "-" * 70
    )

    metric_display_names = {
        "accuracy": "Accuracy",
        "macro_precision": "Macro Precision",
        "macro_recall": "Macro Recall",
        "macro_f1": "Macro F1",
        "weighted_precision": "Weighted Precision",
        "weighted_recall": "Weighted Recall",
        "weighted_f1": "Weighted F1",
        "macro_auprc": "Macro AUPRC",
        "risk_false_positive_rate": "Risk FPR",
    }

    for metric_key, display_name in metric_display_names.items():

        without_value = (
            without_results["metrics"][metric_key]
        )

        with_value = (
            with_results["metrics"][metric_key]
        )

        difference = comparison[metric_key]

        if (
            without_value is None
            or with_value is None
            or difference is None
        ):

            print(
                f"{display_name:<24}"
                f"N/A"
            )

        else:

            print(
                f"{display_name:<24}"
                f"{without_value:>8.4f}"
                f"{with_value:>12.4f}"
                f"{difference:>13.4f}"
            )

    # ---------------------------------------------------------------
    # Save everything
    # ---------------------------------------------------------------

    final_results = {
        "experiment": "SentinelAI provenance comparison",
        "random_seed": 42,
        "test_split": {
            "without_provenance_records": len(test_without),
            "with_provenance_records": len(test_with),
            "trajectory_count": len(
                {
                    str(record["trajectory_id"])
                    for record in test_without
                }
            ),
        },
        "without_provenance": without_results,
        "with_provenance": with_results,
        "comparison": comparison,
    }

    save_results(
        final_results
    )

    print("\nSaved evaluation results:")

    print(
        f"  ✓ {RESULTS_JSON}"
    )

    print(
        f"  ✓ {RESULTS_TEXT}"
    )

    print("\n" + "=" * 70)
    print("✓ STEP 56 COMPLETED")
    print("=" * 70)

    print(
        "\nThe two models have been evaluated on the same "
        "held-out trajectories."
    )

    print(
        "\nIMPORTANT:"
    )

    print(
        "These results are an initial controlled experiment. "
        "They should not yet be interpreted as proof that "
        "provenance causes better security detection."
    )


if __name__ == "__main__":
    main()