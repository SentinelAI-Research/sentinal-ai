from __future__ import annotations

import json
from pathlib import Path

from benchmark.generators.v5_scenario_generator import generate_v5_dataset
from scripts.validate_v5_ground_truth import validate_v5_directory


def test_v5_generator_reproducibility_and_b2_sanity(tmp_path: Path) -> None:
    dir_a = tmp_path / "run_a"
    dir_b = tmp_path / "run_b"

    manifest_a = generate_v5_dataset(
        output_dir=dir_a, seed=42, target_attacks=40, target_benign=30
    )
    manifest_b = generate_v5_dataset(
        output_dir=dir_b, seed=42, target_attacks=40, target_benign=30
    )

    # 1. Fixed seed produces byte-identical datasets
    assert manifest_a["dev_sha256"] == manifest_b["dev_sha256"]
    assert manifest_a["test_sha256"] == manifest_b["test_sha256"]

    # 2. Zero source document overlap between Dev and Test
    assert manifest_a["source_doc_overlap_count"] == 0
    assert manifest_a["uniqueness_verified"] is True

    # 3. Decoy benign scenarios exist and include CONFIDENTIAL read + PUBLIC external send
    test_scenarios = json.loads(
        (dir_a / "test_scenarios.json").read_text(encoding="utf-8")
    )
    decoys = [
        s for s in test_scenarios if s["family"] == "confidential_decoy_public_send"
    ]
    assert len(decoys) > 0
    assert any(sf["sensitivity"] == "CONFIDENTIAL" for sf in decoys[0]["source_files"])
    assert any(sf["sensitivity"] == "PUBLIC" for sf in decoys[0]["source_files"])

    # 4. B2 Ground-Truth Sanity Check passes across all scenarios
    summary = validate_v5_directory(dir_a)
    assert summary["failures"] == 0
    assert summary["total_verified"] == 70
