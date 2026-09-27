from __future__ import annotations

from pathlib import Path

from backend.dataobjects.factory import create_data_object
from backend.provenance.graph import ProvenanceGraph
from ml.scenarios.definitions import long_horizon_leakage_scenario
from ml.scenarios.label_builder import SecurityLabelBuilder
from ml.scenarios.labeled_records import SecurityTrainingRecord
from ml.scenarios.trajectory_executor import ScenarioTrajectoryExecutor


def test_security_label_builder_creates_step_level_training_records() -> None:
    source_path = Path("data/sample/confidential_document.txt")
    source_text = source_path.read_text(encoding="utf-8")

    source_object = create_data_object(
        object_id="test-confidential-source",
        source="controlled-test-source",
        content=source_text,
        sensitivity="CONFIDENTIAL",
        transformation="source_ingestion",
        parents=[],
        created_step="test-source-ingestion",
    )

    scenario = long_horizon_leakage_scenario()

    graph, executed_scenario = ScenarioTrajectoryExecutor.execute(
        scenario=scenario,
        source_object=source_object,
        source_text=source_text,
    )

    records = SecurityLabelBuilder.build(
        scenario=executed_scenario,
        graph=graph,
    )

    assert len(records) == 6

    assert all(isinstance(record, SecurityTrainingRecord) for record in records)

    expected_labels = [
        "SAFE",
        "EARLY_RISK",
        "EARLY_RISK",
        "EARLY_RISK",
        "EARLY_RISK",
        "VIOLATION",
    ]

    actual_labels = [record.label.value for record in records]

    assert actual_labels == expected_labels

    assert records[0].step_number == 1
    assert records[-1].step_number == 6

    assert records[0].total_steps == 6
    assert records[-1].total_steps == 6

    assert records[0].provenance_depth == 1
    assert records[-1].provenance_depth == 6

    assert records[0].transformation_depth == 2
    assert records[-1].transformation_depth == 7

    assert records[0].tool_name == "read_file"
    assert records[-1].tool_name == "send_message"

    assert records[0].output_object_id is not None
    assert records[-1].output_object_id is not None

    assert records[0].timestamp is not None
    assert records[-1].timestamp is not None
