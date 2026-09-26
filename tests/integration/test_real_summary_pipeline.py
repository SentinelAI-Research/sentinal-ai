from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime


def test_real_file_to_summary_preserves_provenance():

    runtime = create_sentinel_runtime()

    registry = DataObjectRegistry(provenance_graph=runtime.provenance_graph)

    loader = FileSourceLoader(registry=registry)

    file_path = Path("data/sample/confidential_document.txt")

    source_object = loader.load_text_file(
        file_path=str(file_path),
        object_id="summary-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="summary-agent")

    # ---------------------------------------------------------
    # Step 1: Read the real file
    # ---------------------------------------------------------

    read_proposal = builder.build(
        step_id="read-step",
        tool_name="read_file",
        arguments={"path": str(file_path)},
        input_data_objects=[source_object.object_id],
    )

    read_decision, read_result = runtime.monitor_service.process(
        proposal=read_proposal,
        destination="INTERNAL",
    )

    assert read_decision.verdict == "ALLOW"
    assert read_result.executed is True
    assert read_result.output

    read_object_id = read_result.output_data_object_id

    assert read_object_id is not None

    # ---------------------------------------------------------
    # Step 2: Summarize the actual file output
    # ---------------------------------------------------------

    summary_proposal = builder.build(
        step_id="summary-step",
        tool_name="summarize",
        arguments={"text": read_result.output},
        input_data_objects=[read_object_id],
    )

    summary_decision, summary_result = runtime.monitor_service.process(
        proposal=summary_proposal,
        destination="INTERNAL",
    )

    assert summary_decision.verdict == "ALLOW"
    assert summary_result.executed is True
    assert summary_result.output

    summary_object_id = summary_result.output_data_object_id

    assert summary_object_id is not None

    # ---------------------------------------------------------
    # Verify provenance
    # ---------------------------------------------------------

    graph = runtime.provenance_graph

    summary_object = graph.get_object(summary_object_id)

    assert summary_object.parents == [read_object_id]

    assert summary_object.transformation == ("summarize")

    assert summary_object.sensitivity == ("CONFIDENTIAL")

    assert graph.get_depth(summary_object_id) == 2
