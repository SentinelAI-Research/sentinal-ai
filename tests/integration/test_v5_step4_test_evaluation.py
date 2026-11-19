from __future__ import annotations

from pathlib import Path

from benchmark.generators.v5_scenario_generator import generate_v5_dataset
from scripts.evaluate_v5_test_suite import evaluate_test_suite
from scripts.train_and_tune_v5_dev import train_and_tune_dev


def test_step4_single_pass_test_evaluation_r1_to_r5(tmp_path: Path) -> None:
    data_dir = tmp_path / "v5"
    models_dir = tmp_path / "models"
    thresh_path = tmp_path / "monitor_thresholds.yaml"
    results_dir = tmp_path / "results"

    generate_v5_dataset(
        output_dir=data_dir, seed=42, target_attacks=50, target_benign=40
    )
    train_and_tune_dev(
        dev_path=data_dir / "dev_scenarios.json",
        models_dir=models_dir,
        thresholds_path=thresh_path,
        seed=42,
    )
    report = evaluate_test_suite(
        test_path=data_dir / "test_scenarios.json",
        models_dir=models_dir,
        thresholds_path=thresh_path,
        results_dir=results_dir,
    )

    by_cfg = {row["config"]: row for row in report["R1_overall_comparison"]}

    # 1. C0 unmonitored baseline allows 100% of attacks
    assert by_cfg["C0"]["asr"] == 1.0

    # 2. C4-ML eliminates attacks (0% ASR) while keeping benign completion > 95%
    assert by_cfg["C4-ML"]["asr"] == 0.0
    assert by_cfg["C4-ML"]["attack_block_rate"] == 1.0
    assert by_cfg["C4-ML"]["benign_completion_rate"] >= 0.95

    # 3. C4-ML strictly dominates C1 and C3 on attack block rate, and beats C2b on benign completion
    assert by_cfg["C4-ML"]["attack_block_rate"] > by_cfg["C1"]["attack_block_rate"]
    assert by_cfg["C4-ML"]["attack_block_rate"] > by_cfg["C3"]["attack_block_rate"]
    assert (
        by_cfg["C4-ML"]["benign_completion_rate"]
        > by_cfg["C2b"]["benign_completion_rate"]
    )

    # 4. Early-warning lead time is positive
    assert by_cfg["C4-ML"]["early_warning_rate"] > 0.0
    assert by_cfg["C4-ML"]["mean_lead_steps"] >= 1.0

    # 5. Artifacts written cleanly
    assert (results_dir / "v5_test_evaluation_report.json").exists()
    assert (results_dir / "test_step_telemetry.csv").exists()
