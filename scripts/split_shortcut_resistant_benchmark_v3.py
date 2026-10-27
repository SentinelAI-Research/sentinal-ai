"""Create a pair-aware train/test split for SentinelAI benchmark V3.

Each matched pair_id is assigned wholly to either train or test. This prevents
its benign and risky variants from being separated across the split.
Outputs are written to data/processed/hard_negative/v3/split/ by default.
Existing V2 data and V3 source data are not modified.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON on line {line_no} of {path}: {exc}"
                    ) from exc
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", default="data/processed/hard_negative/v3/matched_provenance_v3.jsonl"
    )
    parser.add_argument("--output-dir", default="data/processed/hard_negative/v3/split")
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not 0.05 <= args.test_fraction <= 0.40:
        raise SystemExit("--test-fraction must be between 0.05 and 0.40")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input dataset not found: {input_path}")
    rows = read_jsonl(input_path)
    if not rows:
        raise SystemExit("Input dataset is empty")

    required = {
        "pair_id",
        "trajectory_id",
        "step",
        "label",
        "label_encoded",
        "source_family",
        "variant",
    }
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Dataset is missing required fields: {sorted(missing)}")

    pairs: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        pairs[str(row["pair_id"])].append(row)

    # Validate each pair has exactly two trajectories and one benign/risky variant.
    pair_metadata = []
    problems = []
    for pair_id, pair_rows in pairs.items():
        trajectories: dict[str, list[dict]] = defaultdict(list)
        for row in pair_rows:
            trajectories[str(row["trajectory_id"])].append(row)
        variants = {str(r["variant"]) for r in pair_rows}
        if len(trajectories) != 2 or variants != {"benign", "provenance_risky"}:
            problems.append(
                f"{pair_id}: expected two trajectories (benign and provenance_risky)"
            )
            continue
        lengths = {len(steps) for steps in trajectories.values()}
        if len(lengths) != 1:
            problems.append(f"{pair_id}: paired trajectories have different lengths")
        pair_metadata.append((pair_id, str(pair_rows[0]["source_family"])))
    if problems:
        raise SystemExit("Pair validation failed:\n" + "\n".join(problems[:20]))

    # Shuffle pair IDs within source family so both families are represented in each split.
    by_family: dict[str, list[str]] = defaultdict(list)
    for pair_id, family in pair_metadata:
        by_family[family].append(pair_id)

    rng = random.Random(args.seed)
    train_pair_ids: set[str] = set()
    test_pair_ids: set[str] = set()
    for family, ids in sorted(by_family.items()):
        ids = list(ids)
        rng.shuffle(ids)
        n_test = max(1, round(len(ids) * args.test_fraction)) if len(ids) > 1 else 0
        test_pair_ids.update(ids[:n_test])
        train_pair_ids.update(ids[n_test:])

    if train_pair_ids & test_pair_ids:
        raise SystemExit("Internal error: pair IDs overlap between train and test")
    if train_pair_ids | test_pair_ids != set(pairs):
        raise SystemExit("Internal error: some pair IDs were not assigned")

    train_rows = [r for r in rows if str(r["pair_id"]) in train_pair_ids]
    test_rows = [r for r in rows if str(r["pair_id"]) in test_pair_ids]
    # Shuffle rows reproducibly, without changing pair membership.
    random.Random(args.seed + 1).shuffle(train_rows)
    random.Random(args.seed + 2).shuffle(test_rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "train_v3.jsonl", train_rows)
    write_jsonl(out_dir / "test_v3.jsonl", test_rows)

    train_trajs = {str(r["trajectory_id"]) for r in train_rows}
    test_trajs = {str(r["trajectory_id"]) for r in test_rows}
    train_counts = Counter(str(r["label"]) for r in train_rows)
    test_counts = Counter(str(r["label"]) for r in test_rows)
    validation = {
        "benchmark": "SentinelAI Shortcut-Resistant Provenance Evaluation V3",
        "seed": args.seed,
        "test_fraction_requested": args.test_fraction,
        "input": str(input_path),
        "train_rows": len(train_rows),
        "test_rows": len(test_rows),
        "train_pairs": len(train_pair_ids),
        "test_pairs": len(test_pair_ids),
        "train_trajectories": len(train_trajs),
        "test_trajectories": len(test_trajs),
        "train_label_counts": dict(train_counts),
        "test_label_counts": dict(test_counts),
        "train_source_families": dict(
            Counter(str(r["source_family"]) for r in train_rows)
        ),
        "test_source_families": dict(
            Counter(str(r["source_family"]) for r in test_rows)
        ),
        "pair_overlap_count": len(train_pair_ids & test_pair_ids),
        "trajectory_overlap_count": len(train_trajs & test_trajs),
        "pair_integrity_passed": len(train_pair_ids & test_pair_ids) == 0,
        "trajectory_integrity_passed": len(train_trajs & test_trajs) == 0,
        "note": "Matched pair IDs are kept wholly within one split. This is a pair-aware split, not a held-out-template generalization test; the current benchmark has only two source-family labels and uses a fixed action template.",
    }
    (out_dir / "split_validation_v3.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )

    print(f"Input: {input_path}")
    print(f"Output directory: {out_dir}")
    print(
        f"Train: {len(train_rows)} rows | {len(train_trajs)} trajectories | {len(train_pair_ids)} pairs"
    )
    print(
        f"Test:  {len(test_rows)} rows | {len(test_trajs)} trajectories | {len(test_pair_ids)} pairs"
    )
    print(f"Train labels: {dict(train_counts)}")
    print(f"Test labels:  {dict(test_counts)}")
    print(f"Pair overlap: {len(train_pair_ids & test_pair_ids)}")
    print(f"Trajectory overlap: {len(train_trajs & test_trajs)}")
    print("Validation report: " + str(out_dir / "split_validation_v3.json"))
    if (
        not validation["pair_integrity_passed"]
        or not validation["trajectory_integrity_passed"]
    ):
        raise SystemExit("FAIL: train/test leakage detected")
    print("PASS: no matched pair or trajectory appears in both splits.")


if __name__ == "__main__":
    main()
