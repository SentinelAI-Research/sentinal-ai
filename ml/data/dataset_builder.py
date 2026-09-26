import csv
from pathlib import Path

from ml.data.records import TrainingRecord


FIELDNAMES = [
    "trajectory_length",
    "provenance_depth",
    "external_destination",
    "policy_warning",
    "policy_block",
    "transformation_count",
    "label",
    "scenario_id",
    "scenario_family",
    "source_id",
]

"""
Takes training records and creates an actual CSV dataset
"""
def write_dataset(
    records: list[TrainingRecord],
    output_path: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    for record in records:
        record.validate()

    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "trajectory_length": record.trajectory_length,
                    "provenance_depth": record.provenance_depth,
                    "external_destination": record.external_destination,
                    "policy_warning": record.policy_warning,
                    "policy_block": record.policy_block,
                    "transformation_count": record.transformation_count,
                    "label": record.label,
                    "scenario_id": record.scenario_id,
                    "scenario_family": record.scenario_family,
                    "source_id": record.source_id,
                }
            )