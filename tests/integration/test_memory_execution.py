from pathlib import Path

from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.store import TrajectoryStore
from benchmark.runners.execution_service import ScenarioExecutionService
from benchmark.runners.provenance_recorder import ProvenanceRecorder
from benchmark.scenarios.memory_mediated import (
    create_memory_mediated_scenario,
)


def test_memory_path_preserves_provenance():
    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    provenance_recorder = ProvenanceRecorder(
        provenance_graph
    )

    service = ScenarioExecutionService(
        trajectory_store=trajectory_store,
        provenance_recorder=provenance_recorder,
    )

    scenario = create_memory_mediated_scenario()

    file_path = str(
        Path("data/sample/confidential_document.txt")
    )

    execution = service.execute_memory_path(
        scenario=scenario,
        file_path=file_path,
        sensitivity="CONFIDENTIAL",
    )

    assert len(execution.trajectory) == 5

    assert execution.trajectory[0].tool_name == "read_file"
    assert execution.trajectory[1].tool_name == "summarize"
    assert execution.trajectory[2].tool_name == "write_memory"
    assert execution.trajectory[3].tool_name == "read_memory"
    assert execution.trajectory[4].tool_name == "send_message"

    final_object = (
        f"{scenario.scenario_id}-memory-retrieval"
    )

    assert (
        execution.trajectory[-1].input_data_objects
        == [final_object]
    )

    ancestors = provenance_graph.get_ancestors(
        final_object
    )

    assert (
        f"{scenario.scenario_id}-source"
        in ancestors
    )

    assert (
        provenance_graph.get_depth(final_object)
        == 3
    )