from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime


def test_runtime_blocks_confidential_external_action():
    runtime = create_sentinel_runtime()

    registry = DataObjectRegistry(provenance_graph=runtime.provenance_graph)

    loader = FileSourceLoader(registry=registry)

    source = loader.load_text_file(
        file_path=str(Path("data/sample/confidential_document.txt")),
        object_id="security-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="security-test-agent")

    proposal = builder.build(
        step_id="security-block-test",
        tool_name="send_message",
        arguments={
            "recipient": "external-user",
            "message": "Confidential information",
        },
        input_data_objects=[source.object_id],
    )

    decision, execution_result = runtime.monitor_service.process(
        proposal=proposal,
        destination="EXTERNAL",
    )

    assert decision.verdict == "BLOCK"
    assert execution_result.executed is False
    assert execution_result.output_data_object_id is None
