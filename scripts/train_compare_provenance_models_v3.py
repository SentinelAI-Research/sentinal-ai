"""Train and compare baseline vs provenance-aware models on SentinelAI V3.

Reads the pair-aware V3 split and writes all artifacts to models/hard_negative/v3/.
Does not overwrite V2 or the source/split data. Synthetic benchmark only.
"""

from __future__ import annotations
import argparse, json
from pathlib import Path
from collections import Counter
import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

BASELINE_FEATURES = [
    "has_input_object",
    "is_external_action",
    "is_memory_operation",
    "is_transformation",
    "sensitivity_encoded",
    "tool_encoded",
    "transformation_encoded",
]
PROVENANCE_FEATURES = [
    "provenance_depth",
    "transformation_depth",
    "has_provenance",
    "has_sensitive_ancestor",
    "sensitive_ancestor_count",
    "max_ancestor_sensitivity_encoded",
]
LABEL_NAMES = ["SAFE", "EARLY_RISK", "VIOLATION"]


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise SystemExit(
                        f"Invalid JSON in {path}, line {line_no}: {exc}"
                    ) from exc
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train", default="data/processed/hard_negative/v3/split/train_v3.jsonl"
    )
    parser.add_argument(
        "--test", default="data/processed/hard_negative/v3/split/test_v3.jsonl"
    )
    parser.add_argument("--output-dir", default="models/hard_negative/v3")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    train_path, test_path = Path(args.train), Path(args.test)
    if not train_path.exists() or not test_path.exists():
        raise SystemExit("Train/test files not found. Run the V3 split script first.")
    train, test = read_jsonl(train_path), read_jsonl(test_path)
    if not train or not test:
        raise SystemExit("Train or test file is empty.")

    # Basic integrity checks before training.
    train_pairs = {str(r["pair_id"]) for r in train}
    test_pairs = {str(r["pair_id"]) for r in test}
    train_trajs = {str(r["trajectory_id"]) for r in train}
    test_trajs = {str(r["trajectory_id"]) for r in test}
    if train_pairs & test_pairs or train_trajs & test_trajs:
        raise SystemExit(
            "Leakage check failed: pair or trajectory appears in both splits."
        )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_train = np.array([int(r["label_encoded"]) for r in train])
    y_test = np.array([int(r["label_encoded"]) for r in test])
    results = {
        "benchmark": "SentinelAI Shortcut-Resistant Provenance Evaluation V3",
        "synthetic_only": True,
        "seed": args.seed,
        "train_file": str(train_path),
        "test_file": str(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "train_pairs": len(train_pairs),
        "test_pairs": len(test_pairs),
        "pair_overlap_count": len(train_pairs & test_pairs),
        "trajectory_overlap_count": len(train_trajs & test_trajs),
        "models": {},
        "caveat": "Controlled synthetic benchmark. High scores do not establish real-world performance. Inspect feature shortcuts and test unseen scenario templates before drawing generalization claims.",
    }

    for name, features in [
        ("baseline_without_provenance", BASELINE_FEATURES),
        ("provenance_aware", BASELINE_FEATURES + PROVENANCE_FEATURES),
    ]:
        missing = set(features + ["label_encoded"]) - set(train[0])
        if missing:
            raise SystemExit(f"Missing required fields for {name}: {sorted(missing)}")
        X_train = np.asarray(
            [[float(r[f]) for f in features] for r in train], dtype=float
        )
        X_test = np.asarray(
            [[float(r[f]) for f in features] for r in test], dtype=float
        )
        model = HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            l2_regularization=1.0,
            random_state=args.seed,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        acc = float(accuracy_score(y_test, pred))
        macro_f1 = float(f1_score(y_test, pred, average="macro", zero_division=0))
        report = classification_report(
            y_test,
            pred,
            labels=[0, 1, 2],
            target_names=LABEL_NAMES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_test, pred, labels=[0, 1, 2]).tolist()
        model_path = out_dir / f"{name}_v3.joblib"
        joblib.dump(model, model_path)
        model_result = {
            "features": features,
            "feature_count": len(features),
            "accuracy": acc,
            "macro_f1": macro_f1,
            "classification_report": report,
            "confusion_matrix_labels_order": LABEL_NAMES,
            "confusion_matrix": cm,
            "predicted_label_counts": dict(Counter(LABEL_NAMES[int(x)] for x in pred)),
            "model_path": str(model_path),
        }
        # Permutation importance is measured on held-out test rows. It is diagnostic,
        # not proof of causality; correlated features can share importance.
        perm = permutation_importance(
            model,
            X_test,
            y_test,
            scoring="f1_macro",
            n_repeats=5,
            random_state=args.seed,
            n_jobs=1,
        )
        importance = sorted(
            [
                {"feature": f, "mean_macro_f1_decrease": float(m), "std": float(s)}
                for f, m, s in zip(
                    features, perm.importances_mean, perm.importances_std
                )
            ],
            key=lambda x: x["mean_macro_f1_decrease"],
            reverse=True,
        )
        model_result["permutation_importance"] = importance
        results["models"][name] = model_result
        print(f"\n{name}")
        print(f"  Features: {len(features)}")
        print(f"  Accuracy: {acc:.4f}")
        print(f"  Macro F1: {macro_f1:.4f}")
        print(f"  VIOLATION recall: {report['VIOLATION']['recall']:.4f}")
        print(f"  Model: {model_path}")

    base = results["models"]["baseline_without_provenance"]
    prov = results["models"]["provenance_aware"]
    results["difference_provenance_minus_baseline"] = {
        "accuracy_percentage_points": round(
            (prov["accuracy"] - base["accuracy"]) * 100, 4
        ),
        "macro_f1_percentage_points": round(
            (prov["macro_f1"] - base["macro_f1"]) * 100, 4
        ),
        "violation_recall_percentage_points": round(
            (
                prov["classification_report"]["VIOLATION"]["recall"]
                - base["classification_report"]["VIOLATION"]["recall"]
            )
            * 100,
            4,
        ),
    }
    results_path = out_dir / "model_comparison_v3.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("\nComparison report:", results_path)
    print("Pair overlap:", results["pair_overlap_count"])
    print("Trajectory overlap:", results["trajectory_overlap_count"])
    print(
        "Difference (provenance - baseline, percentage points):",
        results["difference_provenance_minus_baseline"],
    )
    print("Reminder: these are synthetic benchmark results, not real-world guarantees.")


if __name__ == "__main__":
    main()
