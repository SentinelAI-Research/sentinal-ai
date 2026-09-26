from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime


def test_confidentiality_survives_transformations():

    runtime = create_sentinel_runtime()

    registry = DataObjectRegistry(provenance_graph=runtime.provenance_graph)

    loader = FileSourceLoader(registry=registry)

    file_path = Path("data/sample/confidential_document.txt")

    source = loader.load_text_file(
        file_path=str(file_path),
        object_id="sensitivity-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="sensitivity-agent")

    # ---------------------------------------------------------
    # Step 1: Read the real file
    # ---------------------------------------------------------

    read_proposal = builder.build(
        step_id="sensitivity-read",
        tool_name="read_file",
        arguments={"path": str(file_path)},
        input_data_objects=[source.object_id],
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

    read_object = runtime.provenance_graph.get_object(read_object_id)

    assert read_object.sensitivity == "CONFIDENTIAL"

    # ---------------------------------------------------------
    # Step 2: Paraphrase the read output
    # ---------------------------------------------------------

    paraphrase_proposal = builder.build(
        step_id="sensitivity-paraphrase",
        tool_name="paraphrase",
        arguments={"text": read_result.output},
        input_data_objects=[read_object_id],
    )

    paraphrase_decision, paraphrase_result = runtime.monitor_service.process(
        proposal=paraphrase_proposal,
        destination="INTERNAL",
    )

    assert paraphrase_decision.verdict == "ALLOW"
    assert paraphrase_result.executed is True
    assert paraphrase_result.output

    paraphrase_object_id = paraphrase_result.output_data_object_id

    assert paraphrase_object_id is not None

    paraphrase_object = runtime.provenance_graph.get_object(paraphrase_object_id)

    # ---------------------------------------------------------
    # Security checks
    # ---------------------------------------------------------

    assert paraphrase_object.sensitivity == "CONFIDENTIAL"

    assert paraphrase_object.parents == [read_object_id]

    assert paraphrase_object.transformation == ("paraphrase")
