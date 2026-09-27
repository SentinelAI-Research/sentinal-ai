"""
Detailed evaluation of SentinelAI provenance hard-negative models.

Compares:
    1. Risk model WITHOUT provenance
    2. Risk model WITH provenance

Evaluation includes:
    - Accuracy
    - Macro precision / recall / F1
    - Weighted precision / recall / F1
    - Per-class metrics
    - Confusion matrices
    - Final-step performance
    - Final-step SAFE vs VIOLATION performance
    - Direct provenance improvement comparison

The same held-out trajectories are used for both models.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data" / "processed" / "hard_negative"
MODEL_DIR = ROOT / "models" / "hard_negative"
RESULT_DIR = ROOT / "results" / "ml_evaluation"

TEST_WITHOUT = DATA_DIR / "test_without_provenance.jsonl"
TEST_WITH = DATA_DIR / "test_with_provenance.jsonl"

MODEL_WITHOUT = (
    MODEL_DIR
    / "security_risk_hard_negative_without_provenance.joblib"
)

MODEL_WITH = (
    MODEL_DIR
    / "security_risk_hard_negative_with_provenance.joblib"
)

LABEL_NAMES = {
    0: "SAFE",
    1: "EARLY_RISK",
    2: "VIOLATION",
}


PROVENANCE_FEATURES = {
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity_encoded",
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

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
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing model: {path}")

    artifact = joblib.load(path)

    if not isinstance(artifact, dict):
        raise ValueError(
            f"Unexpected model artifact format: {path}"
        )

    if "model" not in artifact:
        raise ValueError(
            f"Model artifact does not contain 'model': {path}"
        )

    if "feature_names" not in artifact:
        raise ValueError(
            f"Model artifact does not contain 'feature_names': {path}"
        )

    return artifact


def build_matrix(
    records: list[dict],
    feature_names: list[str],
) -> np.ndarray:

    rows = []

    for record in records:
        row = []

        for feature_name in feature_names:
            value = record.get(feature_name, 0)

            if value is None:
                value = 0

            row.append(float(value))

        rows.append(row)

    return np.asarray(rows, dtype=float)


def evaluate_predictions(
    y_true: list[int],
    y_pred: list[int],
) -> dict:

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=[
            LABEL_NAMES[0],
            LABEL_NAMES[1],
            LABEL_NAMES[2],
        ],
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2],
    )

    return {
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_precision": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_recall": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }


def evaluate_final_step(
    records: list[dict],
    predictions: np.ndarray,
) -> dict:

    final_indices = [
        index
        for index, record in enumerate(records)
        if int(record["step_number"])
        == int(record["total_steps"])
    ]

    if not final_indices:
        raise ValueError("No final-step records found.")

    y_true = [
        int(records[index]["label_encoded"])
        for index in final_indices
    ]

    y_pred = [
        int(predictions[index])
        for index in final_indices
    ]

    result = evaluate_predictions(
        y_true,
        y_pred,
    )

    result["record_count"] = len(final_indices)

    return result


def evaluate_final_safe_vs_violation(
    records: list[dict],
    predictions: np.ndarray,
) -> dict:

    selected_indices = []

    for index, record in enumerate(records):
        is_final = (
            int(record["step_number"])
            == int(record["total_steps"])
        )

        label = int(record["label_encoded"])

        if is_final and label in {0, 2}:
            selected_indices.append(index)

    if not selected_indices:
        raise ValueError(
            "No final-step SAFE/VIOLATION records found."
        )

    y_true = [
        int(records[index]["label_encoded"])
        for index in selected_indices
    ]

    y_pred = [
        int(predictions[index])
        for index in selected_indices
    ]

    result = evaluate_predictions(
        y_true,
        y_pred,
    )

    result["record_count"] = len(selected_indices)

    result["true_class_counts"] = {
        LABEL_NAMES[label]: int(
            sum(1 for value in y_true if value == label)
        )
        for label in [0, 2]
    }

    return result


def print_metrics(
    model_name: str,
    overall: dict,
    final_step: dict,
    final_binary: dict,
) -> None:

    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print(
        f"Accuracy          : "
        f"{overall['accuracy']:.4f}"
    )

    print(
        f"Macro Precision   : "
        f"{overall['macro_precision']:.4f}"
    )

    print(
        f"Macro Recall      : "
        f"{overall['macro_recall']:.4f}"
    )

    print(
        f"Macro F1          : "
        f"{overall['macro_f1']:.4f}"
    )

    print(
        f"Weighted F1       : "
        f"{overall['weighted_f1']:.4f}"
    )

    print()
    print("Overall confusion matrix:")
    print(
        "                Pred SAFE | Pred EARLY_RISK | Pred VIOLATION"
    )

    matrix = overall["confusion_matrix"]

    for index, row in enumerate(matrix):
        print(
            f"True {LABEL_NAMES[index]:<11} "
            f"| {row[0]:>9} "
            f"| {row[1]:>15} "
            f"| {row[2]:>14}"
        )

    print()
    print(
        f"Final-step accuracy : "
        f"{final_step['accuracy']:.4f}"
    )

    print(
        f"Final SAFE/VIOLATION accuracy : "
        f"{final_binary['accuracy']:.4f}"
    )

    print()


def main() -> None:

    print("=" * 70)
    print("SENTINELAI — PROVENANCE HARD-NEGATIVE EVALUATION")
    print("=" * 70)

    test_without = load_jsonl(TEST_WITHOUT)
    test_with = load_jsonl(TEST_WITH)

    if len(test_without) != len(test_with):
        raise ValueError(
            "Test datasets have different numbers of records."
        )

    without_ids = [
        (
            record["trajectory_id"],
            int(record["step_number"]),
        )
        for record in test_without
    ]

    with_ids = [
        (
            record["trajectory_id"],
            int(record["step_number"]),
        )
        for record in test_with
    ]

    if without_ids != with_ids:
        raise ValueError(
            "Test datasets are not aligned."
        )

    print()
    print(
        f"Test records      : {len(test_without)}"
    )

    print(
        f"Test trajectories : "
        f"{len({r['trajectory_id'] for r in test_without})}"
    )

    print()
    print("Test alignment verified.")
    print()

    artifact_without = load_model(MODEL_WITHOUT)
    artifact_with = load_model(MODEL_WITH)

    model_without = artifact_without["model"]
    model_with = artifact_with["model"]

    features_without = artifact_without["feature_names"]
    features_with = artifact_with["feature_names"]

    # ------------------------------------------------------------------
    # WITHOUT PROVENANCE
    # ------------------------------------------------------------------

    X_without = build_matrix(
        test_without,
        features_without,
    )

    y_true_without = [
        int(record["label_encoded"])
        for record in test_without
    ]

    predictions_without = model_without.predict(
        X_without
    )

    overall_without = evaluate_predictions(
        y_true_without,
        predictions_without,
    )

    final_without = evaluate_final_step(
        test_without,
        predictions_without,
    )

    final_binary_without = (
        evaluate_final_safe_vs_violation(
            test_without,
            predictions_without,
        )
    )

    # ------------------------------------------------------------------
    # WITH PROVENANCE
    # ------------------------------------------------------------------

    X_with = build_matrix(
        test_with,
        features_with,
    )

    y_true_with = [
        int(record["label_encoded"])
        for record in test_with
    ]

    predictions_with = model_with.predict(
        X_with
    )

    overall_with = evaluate_predictions(
        y_true_with,
        predictions_with,
    )

    final_with = evaluate_final_step(
        test_with,
        predictions_with,
    )

    final_binary_with = (
        evaluate_final_safe_vs_violation(
            test_with,
            predictions_with,
        )
    )

    # ------------------------------------------------------------------
    # PRINT RESULTS
    # ------------------------------------------------------------------

    print_metrics(
        "MODEL A — WITHOUT PROVENANCE",
        overall_without,
        final_without,
        final_binary_without,
    )

    print_metrics(
        "MODEL B — WITH PROVENANCE",
        overall_with,
        final_with,
        final_binary_with,
    )

    # ------------------------------------------------------------------
    # COMPARISON
    # ------------------------------------------------------------------

    comparison = {
        "without_provenance": {
            "overall": overall_without,
            "final_step": final_without,
            "final_safe_vs_violation": final_binary_without,
            "feature_names": features_without,
        },
        "with_provenance": {
            "overall": overall_with,
            "final_step": final_with,
            "final_safe_vs_violation": final_binary_with,
            "feature_names": features_with,
        },
        "improvement": {
            "accuracy": (
                overall_with["accuracy"]
                - overall_without["accuracy"]
            ),
            "macro_f1": (
                overall_with["macro_f1"]
                - overall_without["macro_f1"]
            ),
            "final_step_accuracy": (
                final_with["accuracy"]
                - final_without["accuracy"]
            ),
            "final_safe_vs_violation_accuracy": (
                final_binary_with["accuracy"]
                - final_binary_without["accuracy"]
            ),
        },
    }

    RESULT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_json = (
        RESULT_DIR
        / "provenance_hard_negative_evaluation.json"
    )

    output_txt = (
        RESULT_DIR
        / "provenance_hard_negative_evaluation.txt"
    )

    output_json.write_text(
        json.dumps(
            comparison,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = []

    lines.append(
        "SENTINELAI — PROVENANCE HARD-NEGATIVE EVALUATION"
    )
    lines.append("=" * 70)
    lines.append("")

    for name, result in [
        (
            "WITHOUT PROVENANCE",
            comparison["without_provenance"],
        ),
        (
            "WITH PROVENANCE",
            comparison["with_provenance"],
        ),
    ]:
        lines.append(name)
        lines.append("-" * 70)

        overall = result["overall"]
        final = result["final_step"]
        binary = result["final_safe_vs_violation"]

        lines.append(
            f"Accuracy: {overall['accuracy']:.4f}"
        )
        lines.append(
            f"Macro Precision: "
            f"{overall['macro_precision']:.4f}"
        )
        lines.append(
            f"Macro Recall: "
            f"{overall['macro_recall']:.4f}"
        )
        lines.append(
            f"Macro F1: "
            f"{overall['macro_f1']:.4f}"
        )
        lines.append(
            f"Weighted F1: "
            f"{overall['weighted_f1']:.4f}"
        )
        lines.append(
            f"Final-step Accuracy: "
            f"{final['accuracy']:.4f}"
        )
        lines.append(
            f"Final SAFE/VIOLATION Accuracy: "
            f"{binary['accuracy']:.4f}"
        )
        lines.append("")

    lines.append("IMPROVEMENT")
    lines.append("-" * 70)

    for metric, value in comparison[
        "improvement"
    ].items():
        lines.append(
            f"{metric}: {value:+.4f}"
        )

    output_txt.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    print("=" * 70)
    print("COMPARISON")
    print("=" * 70)

    print(
        f"Accuracy improvement : "
        f"{comparison['improvement']['accuracy']:+.4f}"
    )

    print(
        f"Macro F1 improvement : "
        f"{comparison['improvement']['macro_f1']:+.4f}"
    )

    print(
        f"Final-step improvement : "
        f"{comparison['improvement']['final_step_accuracy']:+.4f}"
    )

    print(
        f"Final SAFE/VIOLATION improvement : "
        f"{comparison['improvement']['final_safe_vs_violation_accuracy']:+.4f}"
    )

    print()
    print(f"JSON saved to: {output_json}")
    print(f"TXT saved to : {output_txt}")

    print()
    print("=" * 70)
    print("STEP 61 COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()