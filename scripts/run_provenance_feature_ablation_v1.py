"""SentinelAI Split V2 provenance feature ablation.

Reads the existing Split V2 JSONL files and trains fresh models into:
    models/hard_negative/ablation_v1/

It does NOT overwrite the existing Split V2 models or comparison JSON.

Run from repository root:
    python scripts/run_provenance_feature_ablation_v1.py
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "hard_negative" / "split_v2"
OUTPUT_DIR = ROOT / "models" / "hard_negative" / "ablation_v1"

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

SENSITIVE_ANCESTOR_FEATURES = {
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
}

ANCESTOR_SENSITIVITY_FEATURES = {
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity_encoded",
}

PROVENANCE_STRUCTURE_FEATURES = {
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
}

EXCLUDED_FEATURES = {
    # Keep this exclusion list aligned with train_provenance_hard_negative_models_v2.py.
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required Split V2 file not found: {path}\n"
            "Check that the script is run from the repository and Split V2 "
            "data has been generated."
        )

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def validate_records(records: list[dict[str, Any]], name: str) -> None:
    for index, record in enumerate(records):
        for key in ("trajectory_id", "step_number", "label_encoded"):
            if key not in record:
                raise ValueError(f"{name} row {index} missing required key {key!r}")
        try:
            int(record["step_number"])
            int(record["label_encoded"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} row {index} has invalid step/label") from exc


def alignment_key(record: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(record["trajectory_id"]),
        int(record["step_number"]),
        int(record["label_encoded"]),
    )


def validate_split_alignment(
    train_without: list[dict[str, Any]],
    train_with: list[dict[str, Any]],
    test_without: list[dict[str, Any]],
    test_with: list[dict[str, Any]],
) -> dict[str, Any]:
    checks = {
        "train_feature_views_aligned": [alignment_key(row) for row in train_without]
        == [alignment_key(row) for row in train_with],
        "test_feature_views_aligned": [alignment_key(row) for row in test_without]
        == [alignment_key(row) for row in test_with],
    }
    if not all(checks.values()):
        raise ValueError(f"Feature views are not aligned: {checks}")

    train_ids = {str(row["trajectory_id"]) for row in train_with}
    test_ids = {str(row["trajectory_id"]) for row in test_with}
    overlap = sorted(train_ids & test_ids)
    if overlap:
        raise ValueError(
            f"Train/test trajectory overlap detected ({len(overlap)} IDs): "
            f"{overlap[:10]}"
        )

    if [int(r["label_encoded"]) for r in train_without] != [
        int(r["label_encoded"]) for r in train_with
    ]:
        raise ValueError("Training labels differ between feature views.")
    if [int(r["label_encoded"]) for r in test_without] != [
        int(r["label_encoded"]) for r in test_with
    ]:
        raise ValueError("Test labels differ between feature views.")

    return {
        **checks,
        "train_test_trajectory_overlap_count": len(overlap),
        "train_rows": len(train_with),
        "test_rows": len(test_with),
        "train_trajectories": len(train_ids),
        "test_trajectories": len(test_ids),
    }


def discover_numeric_features(records: list[dict[str, Any]]) -> list[str]:
    keys = set().union(*(row.keys() for row in records))
    features: list[str] = []

    for key in sorted(keys):
        if key in EXCLUDED_FEATURES:
            continue
        values = [row.get(key) for row in records if row.get(key) is not None]
        if not values:
            continue
        # Match the V2 training script: booleans are not numeric features.
        if all(
            isinstance(value, (int, float)) and not isinstance(value, bool)
            for value in values
        ):
            features.append(key)

    if not features:
        raise ValueError("No numeric model features discovered.")
    return features


def build_matrix(records: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    rows: list[list[float]] = []
    for row_index, record in enumerate(records):
        values: list[float] = []
        for feature in feature_names:
            value = record.get(feature, 0)
            if value is None:
                value = 0
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    f"Non-numeric value for feature {feature!r}, row {row_index}: "
                    f"{value!r}"
                )
            values.append(float(value))
        rows.append(values)
    return np.asarray(rows, dtype=float)


def labels(records: list[dict[str, Any]]) -> np.ndarray:
    return np.asarray([int(row["label_encoded"]) for row in records], dtype=int)


def discover_label_names(records: list[dict[str, Any]]) -> dict[int, str]:
    names: dict[int, set[str]] = defaultdict(set)
    for row in records:
        encoded = int(row["label_encoded"])
        raw_label = row.get("label")
        if raw_label is not None:
            names[encoded].add(str(raw_label))

    ambiguous = {key: sorted(value) for key, value in names.items() if len(value) > 1}
    if ambiguous:
        raise ValueError(f"Label encoding is inconsistent: {ambiguous}")

    return {
        key: next(iter(value)) if value else f"CLASS_{key}"
        for key, value in names.items()
    }


def label_feature_audit(
    records: list[dict[str, Any]], feature_names: list[str], label_names: dict[int, str]
) -> dict[str, Any]:
    """Summarize feature distributions by target class to flag possible shortcuts."""
    audit: dict[str, Any] = {}
    y = labels(records)

    for feature in feature_names:
        by_class: dict[str, Any] = {}
        for class_id in sorted(set(y.tolist())):
            values = [
                row.get(feature, 0) if row.get(feature) is not None else 0
                for row, target in zip(records, y)
                if target == class_id
            ]
            counts = Counter(str(value) for value in values)
            by_class[label_names.get(class_id, str(class_id))] = {
                "rows": len(values),
                "unique_values": len(counts),
                "value_counts": dict(sorted(counts.items())),
            }

        # Flag features whose observed value sets are disjoint between any pair
        # of classes. This is a warning heuristic, not proof of causal leakage.
        value_sets = {
            class_name: set(details["value_counts"])
            for class_name, details in by_class.items()
        }
        class_names = list(value_sets)
        disjoint_pairs = []
        for i, first in enumerate(class_names):
            for second in class_names[i + 1 :]:
                if value_sets[first].isdisjoint(value_sets[second]):
                    disjoint_pairs.append([first, second])

        audit[feature] = {
            "by_class": by_class,
            "disjoint_value_set_class_pairs": disjoint_pairs,
            "possible_shortcut_warning": bool(disjoint_pairs),
        }
    return audit


def feature_sets(
    all_features: list[str], baseline_features: list[str]
) -> dict[str, list[str]]:
    available = set(all_features)
    baseline = [name for name in baseline_features if name in available]
    provenance = [name for name in all_features if name in PROVENANCE_FEATURES]

    variants: dict[str, list[str]] = {
        "baseline_without_provenance": baseline,
        "full_provenance": list(all_features),
        "drop_has_sensitive_ancestor": [
            f for f in all_features if f != "has_sensitive_ancestor"
        ],
        "drop_sensitive_ancestor_count": [
            f for f in all_features if f != "sensitive_ancestor_count"
        ],
        "drop_both_sensitive_ancestor_features": [
            f for f in all_features if f not in SENSITIVE_ANCESTOR_FEATURES
        ],
        "drop_ancestor_sensitivity_group": [
            f for f in all_features if f not in ANCESTOR_SENSITIVITY_FEATURES
        ],
        "drop_provenance_structure_group": [
            f for f in all_features if f not in PROVENANCE_STRUCTURE_FEATURES
        ],
        "drop_all_provenance": [
            f for f in all_features if f not in PROVENANCE_FEATURES
        ],
    }

    # Keep only existing features, preserve stable order, and reject accidental
    # empty or duplicate feature lists.
    cleaned: dict[str, list[str]] = {}
    for name, features in variants.items():
        ordered = list(dict.fromkeys(f for f in features if f in available))
        if not ordered:
            raise ValueError(f"Ablation {name!r} has no features.")
        cleaned[name] = ordered

    return cleaned


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_ids: list[int],
    label_names: dict[int, str],
) -> dict[str, Any]:
    report = classification_report(
        y_true,
        y_pred,
        labels=class_ids,
        target_names=[label_names.get(c, f"CLASS_{c}") for c in class_ids],
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=class_ids)

    per_class = {}
    for class_id in class_ids:
        name = label_names.get(class_id, f"CLASS_{class_id}")
        values = report[name]
        per_class[name] = {
            "precision": values["precision"],
            "recall": values["recall"],
            "f1": values["f1-score"],
            "support": int(values["support"]),
        }

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=class_ids, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=class_ids, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true, y_pred, labels=class_ids, average="macro", zero_division=0
            )
        ),
        "per_class": per_class,
        "confusion_matrix": {
            "labels": [label_names.get(c, f"CLASS_{c}") for c in class_ids],
            "rows_true_columns_predicted": cm.tolist(),
        },
        "classification_report": report,
    }


def main() -> None:
    print("=" * 78)
    print("SENTINELAI — PROVENANCE FEATURE ABLATION V1")
    print("=" * 78)
    print(f"Input split: {SPLIT_DIR}")
    print(f"Output dir:  {OUTPUT_DIR}")
    print("Existing Split V2 model/result files will not be modified.")

    train_without = load_jsonl(TRAIN_WITHOUT)
    test_without = load_jsonl(TEST_WITHOUT)
    train_with = load_jsonl(TRAIN_WITH)
    test_with = load_jsonl(TEST_WITH)

    for name, rows in (
        ("train_without", train_without),
        ("test_without", test_without),
        ("train_with", train_with),
        ("test_with", test_with),
    ):
        validate_records(rows, name)

    integrity = validate_split_alignment(
        train_without, train_with, test_without, test_with
    )
    y_train = labels(train_with)
    y_test = labels(test_with)
    class_ids = sorted(set(y_train.tolist()) | set(y_test.tolist()))
    label_names = discover_label_names(train_with + test_with)

    if set(y_train.tolist()) != set(y_test.tolist()):
        raise ValueError("Train/test label classes differ.")

    train_features = discover_numeric_features(train_with)
    test_features = discover_numeric_features(test_with)
    if train_features != test_features:
        raise ValueError(
            "Train/test numeric feature schema differs.\n"
            f"Train: {train_features}\nTest: {test_features}"
        )

    baseline_features = discover_numeric_features(train_without)
    baseline_test_features = discover_numeric_features(test_without)
    if baseline_features != baseline_test_features:
        raise ValueError("Baseline train/test feature schema differs.")

    # Label/ID leakage guard: these columns must never be model inputs.
    forbidden = {"label", "label_encoded"} & set(train_features)
    if forbidden:
        raise ValueError(
            f"Target leakage: label fields discovered as features: {forbidden}"
        )

    variants = feature_sets(train_features, baseline_features)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Avoid silently overwriting an earlier ablation run.
    summary_path = OUTPUT_DIR / "ablation_summary_v1.json"
    if summary_path.exists():
        raise FileExistsError(
            f"{summary_path} already exists. Preserve it and choose a new output "
            "folder/version before rerunning."
        )

    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_files": {
            "train_without": str(TRAIN_WITHOUT.relative_to(ROOT)),
            "test_without": str(TEST_WITHOUT.relative_to(ROOT)),
            "train_with": str(TRAIN_WITH.relative_to(ROOT)),
            "test_with": str(TEST_WITH.relative_to(ROOT)),
        },
        "split_integrity": integrity,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_config": CLASSIFIER_CONFIG,
        "class_id_to_name": {str(k): v for k, v in sorted(label_names.items())},
        "train_label_counts": dict(
            Counter(label_names.get(int(y), str(y)) for y in y_train)
        ),
        "test_label_counts": dict(
            Counter(label_names.get(int(y), str(y)) for y in y_test)
        ),
        "all_provenance_features_detected": sorted(
            set(train_features) & PROVENANCE_FEATURES
        ),
        "excluded_feature_names": sorted(EXCLUDED_FEATURES),
        "feature_sets": variants,
        "feature_label_shortcut_audit_on_training_data": label_feature_audit(
            train_with, train_features, label_names
        ),
        "feature_label_shortcut_audit_on_test_data": label_feature_audit(
            test_with, train_features, label_names
        ),
        "warning": (
            "Feature-label distribution warnings indicate possible shortcuts, "
            "not proof of causal leakage. The benchmark is synthetic/controlled; "
            "scores are not real-world security guarantees."
        ),
    }
    write_json(OUTPUT_DIR / "ablation_audit_v1.json", audit)

    results: dict[str, Any] = {}
    model = None

    for variant_name, feature_names in variants.items():
        print("\n" + "-" * 78)
        print(f"Running: {variant_name} ({len(feature_names)} features)")
        print("-" * 78)

        if variant_name == "baseline_without_provenance":
            train_rows, test_rows = train_without, test_without
        else:
            train_rows, test_rows = train_with, test_with

        X_train = build_matrix(train_rows, feature_names)
        X_test = build_matrix(test_rows, feature_names)
        y_train_variant = labels(train_rows)
        y_test_variant = labels(test_rows)

        if X_train.shape[1] != len(feature_names) or X_test.shape[1] != len(
            feature_names
        ):
            raise ValueError(f"Matrix/feature mismatch in {variant_name}")

        model = HistGradientBoostingClassifier(**CLASSIFIER_CONFIG)
        model.fit(X_train, y_train_variant)
        predictions = model.predict(X_test)

        metrics = calculate_metrics(y_test_variant, predictions, class_ids, label_names)
        results[variant_name] = {
            "feature_count": len(feature_names),
            "features": feature_names,
            "train_rows": len(train_rows),
            "test_rows": len(test_rows),
            "train_trajectories": len({str(r["trajectory_id"]) for r in train_rows}),
            "test_trajectories": len({str(r["trajectory_id"]) for r in test_rows}),
            "metrics": metrics,
        }

        # Save each model separately within the ablation directory only.
        import joblib

        model_path = OUTPUT_DIR / f"{variant_name}.joblib"
        joblib.dump(
            {
                "model": model,
                "feature_names": feature_names,
                "classifier_config": CLASSIFIER_CONFIG,
                "variant": variant_name,
            },
            model_path,
        )
        write_json(
            OUTPUT_DIR / f"{variant_name}_metrics.json",
            results[variant_name],
        )

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1:  {metrics['macro_f1']:.4f}")
        print("Per-class metrics:")
        for class_name, values in metrics["per_class"].items():
            print(
                f"  {class_name:12s} "
                f"P={values['precision']:.4f} "
                f"R={values['recall']:.4f} "
                f"F1={values['f1']:.4f} "
                f"support={values['support']}"
            )

    full = results["full_provenance"]["metrics"]
    compact = {}
    for name, item in results.items():
        metrics = item["metrics"]
        compact[name] = {
            "feature_count": item["feature_count"],
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "accuracy_delta_vs_full_provenance": metrics["accuracy"] - full["accuracy"],
            "macro_f1_delta_vs_full_provenance": metrics["macro_f1"] - full["macro_f1"],
            "per_class": metrics["per_class"],
            "confusion_matrix": metrics["confusion_matrix"],
        }

    summary = {
        "experiment": "sentinelai_provenance_feature_ablation_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "split_integrity": integrity,
        "classifier": "HistGradientBoostingClassifier",
        "classifier_config": CLASSIFIER_CONFIG,
        "class_id_to_name": {str(k): v for k, v in sorted(label_names.items())},
        "results": compact,
        "interpretation_note": (
            "Compare feature ablations on the same held-out Split V2 rows. "
            "Large score drops after removing a feature indicate model dependence, "
            "not necessarily useful causal signal. Inspect ablation_audit_v1.json "
            "for features whose values separate labels unusually cleanly."
        ),
    }
    write_json(summary_path, summary)

    # Permutation importance on the full provenance model, computed only after
    # training on the same fixed split. Use macro-F1 scoring to foreground classes.
    print("\nComputing held-out permutation importance for full_provenance...")
    full_features = variants["full_provenance"]
    full_model = HistGradientBoostingClassifier(**CLASSIFIER_CONFIG)
    full_model.fit(build_matrix(train_with, full_features), labels(train_with))
    permutation = permutation_importance(
        full_model,
        build_matrix(test_with, full_features),
        labels(test_with),
        scoring="f1_macro",
        n_repeats=10,
        random_state=42,
        n_jobs=1,
    )
    importance_rows = sorted(
        [
            {
                "feature": feature,
                "mean_macro_f1_decrease": float(mean),
                "std": float(std),
            }
            for feature, mean, std in zip(
                full_features, permutation.importances_mean, permutation.importances_std
            )
        ],
        key=lambda row: row["mean_macro_f1_decrease"],
        reverse=True,
    )
    write_json(
        OUTPUT_DIR / "full_provenance_permutation_importance_v1.json",
        {
            "method": "sklearn.inspection.permutation_importance",
            "scoring": "f1_macro",
            "n_repeats": 10,
            "random_state": 42,
            "results": importance_rows,
            "caveat": (
                "Correlated features can share or mask importance. Treat this as "
                "diagnostic evidence, not causal attribution."
            ),
        },
    )

    print("\n" + "=" * 78)
    print("ABLATION COMPLETE")
    print("=" * 78)
    print(f"Summary: {summary_path}")
    print(f"Audit:   {OUTPUT_DIR / 'ablation_audit_v1.json'}")
    print(
        f"Permutation importance: {OUTPUT_DIR / 'full_provenance_permutation_importance_v1.json'}"
    )
    print("\nQuick comparison:")
    print(f"{'Variant':42s} {'Features':>8s} {'Accuracy':>10s} {'Macro F1':>10s}")
    for name, item in results.items():
        metrics = item["metrics"]
        print(
            f"{name:42s} {item['feature_count']:8d} "
            f"{metrics['accuracy']:10.4f} {metrics['macro_f1']:10.4f}"
        )
    print("\nExisting models and results were not overwritten.")


if __name__ == "__main__":
    main()
