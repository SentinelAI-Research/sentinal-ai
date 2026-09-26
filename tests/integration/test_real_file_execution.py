from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime
from backend.provenance.graph import ProvenanceGraph


def test_real_file_read_creates_provenance_output():

    runtime = create_sentinel_runtime()

    graph: ProvenanceGraph = runtime.provenance_graph

    registry = DataObjectRegistry(provenance_graph=graph)

    loader = FileSourceLoader(registry=registry)

    file_path = Path("data/sample/confidential_document.txt")

    source_object = loader.load_text_file(
        file_path=str(file_path),
        object_id="real-file-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="integration-agent")

    proposal = builder.build(
        step_id="read-file-step",
        tool_name="read_file",
        arguments={"path": str(file_path)},
        input_data_objects=[source_object.object_id],
    )

    decision, execution_result = runtime.monitor_service.process(
        proposal=proposal,
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"

    assert execution_result.executed is True

    assert execution_result.output

    assert execution_result.output_data_object_id == "read-file-step-output"

    output_object = graph.get_object("read-file-step-output")

    assert output_object.parents == ["real-file-source"]

    assert output_object.transformation == ("read_file")

    assert output_object.sensitivity == ("CONFIDENTIAL")

    assert graph.get_depth("read-file-step-output") == 1
