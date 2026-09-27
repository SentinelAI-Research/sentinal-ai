from __future__ import annotations

import sys
from pathlib import Path

# Add project root to Python import path.
ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data.enron_security_dataset import EnronSecurityDatasetBuilder


ARCHIVE_PATH = (
    ROOT
    / "data"
    / "raw"
    / "enron"
    / "enron_mail_20150507.tar.gz"
)

OUTPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "enron"
    / "enron_security_experiment.jsonl"
)

TOTAL_TRAJECTORIES = 100
STEPS_PER_TRAJECTORY = 6


def main() -> None:
    print("=" * 60)
    print("SENTINELAI — ENRON EXPERIMENT DATASET")
    print("=" * 60)
    print()
    print(f"Archive: {ARCHIVE_PATH}")
    print(f"Output:  {OUTPUT_PATH}")
    print()
    print(
        f"Target: {TOTAL_TRAJECTORIES} trajectories "
        f"× {STEPS_PER_TRAJECTORY} steps "
        f"= {TOTAL_TRAJECTORIES * STEPS_PER_TRAJECTORY} records"
    )
    print()

    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Enron archive not found: {ARCHIVE_PATH}"
        )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating dataset...")
    print(
        "Each trajectory uses isolated temporary memory."
    )
    print()

    generated = EnronSecurityDatasetBuilder.build_from_archive(
        archive_path=ARCHIVE_PATH,
        output_path=OUTPUT_PATH,
        max_records=TOTAL_TRAJECTORIES,
    )

    print()
    print("=" * 60)
    print("ENRON EXPERIMENT DATASET CREATED")
    print("=" * 60)
    print(f"Trajectory sources : {TOTAL_TRAJECTORIES}")
    print(f"Trajectory steps   : {generated}")
    print(
        f"Expected steps     : "
        f"{TOTAL_TRAJECTORIES * STEPS_PER_TRAJECTORY}"
    )
    print(f"Output             : {OUTPUT_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()