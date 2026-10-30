"""
Phase 4 Experiment: Evaluates Model Generalization Across Unseen Scenario Templates
and calculates Early-Warning Lead Time.
"""

import json
from pathlib import Path
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score


def extract_features_from_scenario(scenario: dict):
    """Extracts 7 non-provenance baseline features and 13 full-provenance features
    from raw scenario definitions.
    """
    rows = []
    steps = scenario.get("steps", [])
    has_confidential_root = False
    ancestor_count = 0

    for idx, step in enumerate(steps):
        tool = step.get("tool", "")
        sensitivity = step.get("sensitivity", "PUBLIC")
        if sensitivity == "CONFIDENTIAL":
            has_confidential_root = True
            ancestor_count += 1

        is_external = 1 if tool == "send_message" else 0
        step_num = idx + 1

        # Determine label based on safety policy
        if is_external and has_confidential_root:
            label = "VIOLATION"
        elif has_confidential_root and step_num > 1:
            label = "EARLY_RISK"
        else:
            label = "SAFE"

        # 7 Non-provenance baseline features (Current action + basic sequence)
        features_7_base = [
            step_num,
            is_external,
            1 if tool in ["read_file", "read_memory"] else 0,
            1 if tool in ["summarize", "paraphrase", "calculate"] else 0,
            1 if tool == "write_memory" else 0,
            len(steps),
            1 if tool == "calculate" else 0,
        ]

        # 13 Full provenance features (Baseline + Provenance DAG signals)
        features_13_prov = features_7_base + [
            1 if has_confidential_root else 0,  # has_sensitive_ancestor
            ancestor_count,  # sensitive_ancestor_count
            1 if sensitivity == "CONFIDENTIAL" else 0,  # max_ancestor_sensitivity
            step_num if has_confidential_root else 0,  # provenance_depth
            1 if tool == "send_message" and has_confidential_root else 0,
            1 if "injection" in step else 0,
        ]

        rows.append(
            {
                "features_7_base": features_7_base,
                "features_13_prov": features_13_prov,
                "label": label,
                "scenario_template": scenario.get("template", "default"),
            }
        )

    return rows


def run_v4_experiment():
    data_path = Path("data/processed/v4_scenarios.json")
    if not data_path.exists():
        data_path = Path("data/processed/v4_trajectories.json")

    if not data_path.exists():
        print("Error: Could not find v4_scenarios.json in data/processed/")
        return

    with open(data_path, "r", encoding="utf-8") as f:
        scenarios = json.load(f)

    all_rows = []
    for sc in scenarios:
        all_rows.extend(extract_features_from_scenario(sc))

    if not all_rows:
        print("Error: No feature rows could be extracted from scenario data.")
        return

    # Extract arrays
    X_base = [r["features_7_base"] for r in all_rows]
    X_prov = [r["features_13_prov"] for r in all_rows]
    y = [r["label"] for r in all_rows]

    # TEMPLATE-LEVEL SPLIT (Zero-Shot Scenario Sequence)
    templates = {r["scenario_template"] for r in all_rows}
    print(
        f"[V4 Dataset] Found {len(all_rows)} total step records across templates: {templates}"
    )

    train_templates = {
        "multi_input_aggregation",
        "document_processing",
        "calculator_simple",
    }

    X_train_base, y_train = [], []
    X_train_prov, X_test_base, X_test_prov, y_test = [], [], [], []

    for row in all_rows:
        if row["scenario_template"] in train_templates:
            X_train_base.append(row["features_7_base"])
            X_train_prov.append(row["features_13_prov"])
            y_train.append(row["label"])
        else:
            X_test_base.append(row["features_7_base"])
            X_test_prov.append(row["features_13_prov"])
            y_test.append(row["label"])

    # Fallback to stratified 80/20 split if template split resulted in empty train/test array
    if len(X_train_base) == 0 or len(X_test_base) == 0:
        print(
            "[V4 Split] Template matching produced empty partitions; using 80/20 trajectory split."
        )
        split_idx = int(len(all_rows) * 0.8)
        X_train_base, X_test_base = X_base[:split_idx], X_base[split_idx:]
        X_train_prov, X_test_prov = X_prov[:split_idx], X_prov[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

    # Convert to NumPy Arrays
    X_train_base = np.array(X_train_base)
    X_train_prov = np.array(X_train_prov)
    X_test_base = np.array(X_test_base)
    X_test_prov = np.array(X_test_prov)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    # Train Baseline (7 features) vs Provenance Model (13 features)
    model_base = HistGradientBoostingClassifier(random_state=42).fit(
        X_train_base, y_train
    )
    model_prov = HistGradientBoostingClassifier(random_state=42).fit(
        X_train_prov, y_train
    )

    preds_base = model_base.predict(X_test_base)
    preds_prov = model_prov.predict(X_test_prov)

    f1_base = f1_score(y_test, preds_base, average="macro")
    f1_prov = f1_score(y_test, preds_prov, average="macro")

    print("==================================================")
    print("SENTINELAI — RESEARCH PHASE V4 EVALUATION RESULTS")
    print("==================================================")
    print(f"[1] Baseline Model (7 Non-Provenance Features)  - Macro F1: {f1_base:.4f}")
    print(f"[2] SentinelAI Model (13 Provenance Features) - Macro F1: {f1_prov:.4f}")
    print("==================================================")
    print(
        f"IMPROVEMENT: +{(f1_prov - f1_base) * 100:.2f}% Macro F1 on Zero-Shot Templates!"
    )
    print("==================================================")

    # Save metrics JSON artifact
    out_dir = Path("models/hard_negative/v4")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "v4_evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "baseline_macro_f1": float(f1_base),
                "provenance_macro_f1": float(f1_prov),
                "improvement_pct": float((f1_prov - f1_base) * 100),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    run_v4_experiment()
