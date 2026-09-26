from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.store import TrajectoryStore
from benchmark.runners.execution_service import ScenarioExecutionService
from benchmark.runners.provenance_recorder import ProvenanceRecorder
from benchmark.scenarios.summary_leakage import (
    create_summary_leakage_scenario,
)


def test_summary_execution_creates_trajectory_and_provenance():
    trajectory_store = TrajectoryStore()

    provenance_graph = ProvenanceGraph()

    provenance_recorder = ProvenanceRecorder(
        provenance_graph
    )

    service = ScenarioExecutionService(
        trajectory_store=trajectory_store,
        provenance_recorder=provenance_recorder,
    )

    scenario = create_summary_leakage_scenario()

    file_path = str(
        Path("data/sample/confidential_document.txt")
    )

    execution = service.execute_summary(
        scenario=scenario,
        file_path=file_path,
        sensitivity="CONFIDENTIAL",
    )

    assert len(execution.trajectory) == 2

    assert (
        execution.trajectory[0].tool_name
        == "read_file"
    )

    assert (
        execution.trajectory[1].tool_name
        == "summarize"
    )

    assert provenance_graph.get_parents(
        f"{scenario.scenario_id}-summary"
    ) == [f"{scenario.scenario_id}-source"]

    assert provenance_graph.get_depth(
        f"{scenario.scenario_id}-summary"
    ) == 1