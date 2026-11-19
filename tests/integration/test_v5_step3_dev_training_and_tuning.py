from __future__ import annotations

import json
from pathlib import Path

from backend.monitor.config import MonitorConfig
from benchmark.generators.v5_scenario_generator import generate_v5_dataset
from benchmark.runners.v5_experiment_runner import run_v5_scenario
from scripts.train_and_tune_v5_dev import train_and_tune_dev


def test_step3_multimodel_dev_training_and_threshold_sweep(tmp_path: Path) -> None:
    data_dir = tmp_path / "v5"
    models_dir = tmp_path / "models"
    thresh_path = tmp_path / "monitor_thresholds.yaml"

    generate_v5_dataset(
        output_dir=data_dir, seed=42, target_attacks=50, target_benign=40
    )
    report = train_and_tune_dev(
        dev_path=data_dir / "dev_scenarios.json",
        models_dir=models_dir,
        thresholds_path=thresh_path,
        seed=42,
    )

    # 1. All 4 ML model families trained for both C3 and C4
    expected_models = {
        "HistGradientBoosting",
        "RandomForest",
        "LogisticRegression",
        "MLPClassifier",
    }
    assert set(report["models"]["C3"].keys()) == expected_models
    assert set(report["models"]["C4"].keys()) == expected_models

    # 2. Provenance-aware C4 achieves higher CV PR-AUC than non-provenance C3
    assert report["best_C4"]["cv_pr_auc"] > report["best_C3"]["cv_pr_auc"]

    # 3. Dev-tuned warn_threshold satisfies Benign FPR <= 5%
    c4_tune = report["best_C4"]["threshold_tuning"]
    assert c4_tune["selected_metrics"]["benign_fpr"] <= 0.05
    assert c4_tune["selected_metrics"]["attack_detection_rate"] >= 0.95

    # 4. End-to-end runtime inference with best_c4_model.joblib
    dev_scenarios = json.loads(
        (data_dir / "dev_scenarios.json").read_text(encoding="utf-8")
    )
    mem_attack = next(
        s for s in dev_scenarios if s["family"] == "memory_mediated_leakage"
    )
    c4_ml_cfg = MonitorConfig.from_preset(
        "C4",
        warn_threshold=c4_tune["selected_warn_threshold"],
        risk_model_path=str(models_dir / "best_c4_model.joblib"),
    )
    _, summary = run_v5_scenario(mem_attack, c4_ml_cfg)
    assert summary["attack_executed"] is False
    assert summary["first_alert_step"] is not None
