from __future__ import annotations

import json
from pathlib import Path

from backend.monitor.config import MonitorConfig
from benchmark.generators.v5_scenario_generator import generate_v5_dataset
from benchmark.runners.v5_experiment_runner import run_v5_scenario


def test_step2_monitor_configs_and_early_warning_fusion(tmp_path: Path) -> None:
    generate_v5_dataset(
        output_dir=tmp_path, seed=42, target_attacks=35, target_benign=25
    )
    dev_scenarios = json.loads(
        (tmp_path / "dev_scenarios.json").read_text(encoding="utf-8")
    )

    mem_attack = next(
        s for s in dev_scenarios if s["family"] == "memory_mediated_leakage"
    )
    decoy_benign = next(
        s for s in dev_scenarios if s["family"] == "confidential_decoy_public_send"
    )

    # 1. C0 (No monitor): attack executes completely without alerts
    _, sum_c0 = run_v5_scenario(mem_attack, MonitorConfig.from_preset("C0"))
    assert sum_c0["attack_executed"] is True
    assert sum_c0["first_alert_step"] is None

    # 2. C1 (Action-only, no inheritance): misses multi-hop memory leak
    _, sum_c1 = run_v5_scenario(mem_attack, MonitorConfig.from_preset("C1"))
    assert sum_c1["attack_executed"] is True

    # 3. C4 (Full SentinelAI): issues early WARN before violation_step and BLOCKs send_message
    steps_c4, sum_c4 = run_v5_scenario(mem_attack, MonitorConfig.from_preset("C4"))
    assert sum_c4["attack_executed"] is False
    assert sum_c4["first_alert_step"] is not None
    assert sum_c4["first_alert_step"] < mem_attack["violation_step"]
    assert steps_c4[-1]["verdict"] == "BLOCK"

    # 4. Decoy Benign Comparison: C2b false-blocks, C4 safely allows
    _, decoy_c2b = run_v5_scenario(decoy_benign, MonitorConfig.from_preset("C2b"))
    _, decoy_c4 = run_v5_scenario(decoy_benign, MonitorConfig.from_preset("C4"))
    assert decoy_c2b["completed"] is False
    assert decoy_c2b["first_alert_step"] is not None
    assert decoy_c4["completed"] is True
    assert decoy_c4["first_alert_step"] is None
