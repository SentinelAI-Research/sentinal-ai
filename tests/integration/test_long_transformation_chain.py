from pathlib import Path

from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.dataobjects.source_loader import FileSourceLoader
from backend.monitor.factory import create_sentinel_runtime


def test_long_transformation_chain():

    runtime = create_sentinel_runtime()

    registry = DataObjectRegistry(provenance_graph=runtime.provenance_graph)

    loader = FileSourceLoader(registry=registry)

    file_path = Path("data/sample/confidential_document.txt")

    # ---------------------------------------------------------
    # Create the provenance root from the real file
    # ---------------------------------------------------------

    source = loader.load_text_file(
        file_path=str(file_path),
        object_id="chain-source",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="chain-agent")

    # =========================================================
    # STEP 1 — READ
    # =========================================================

    read_proposal = builder.build(
        step_id="chain-read",
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

    read_id = read_result.output_data_object_id

    assert read_id is not None

    # =========================================================
    # STEP 2 — SUMMARIZE
    # =========================================================

    summary_proposal = builder.build(
        step_id="chain-summary",
        tool_name="summarize",
        arguments={"text": read_result.output},
        input_data_objects=[read_id],
    )

    summary_decision, summary_result = runtime.monitor_service.process(
        proposal=summary_proposal,
        destination="INTERNAL",
    )

    assert summary_decision.verdict == "ALLOW"
    assert summary_result.executed is True
    assert summary_result.output

    summary_id = summary_result.output_data_object_id

    assert summary_id is not None

    # =========================================================
    # STEP 3 — PARAPHRASE
    # =========================================================

    paraphrase_proposal = builder.build(
        step_id="chain-paraphrase",
        tool_name="paraphrase",
        arguments={"text": summary_result.output},
        input_data_objects=[summary_id],
    )

    paraphrase_decision, paraphrase_result = runtime.monitor_service.process(
        proposal=paraphrase_proposal,
        destination="INTERNAL",
    )

    assert paraphrase_decision.verdict == "ALLOW"
    assert paraphrase_result.executed is True
    assert paraphrase_result.output

    paraphrase_id = paraphrase_result.output_data_object_id

    assert paraphrase_id is not None

    # =========================================================
    # VERIFY FINAL PROVENANCE
    # =========================================================

    graph = runtime.provenance_graph

    final_object = graph.get_object(paraphrase_id)

    assert final_object.sensitivity == ("CONFIDENTIAL")

    assert final_object.parents == [summary_id]

    assert final_object.transformation == ("paraphrase")

    # The final object is three edges away from
    # the original source.
    assert graph.get_depth(paraphrase_id) == 3

    # Three transformations occurred:
    #
    # read_file
    # summarize
    # paraphrase
    #
    assert graph.get_transformation_depth(paraphrase_id) == 3

    # The complete ancestor chain should contain
    # all previous objects.
    ancestors = graph.get_ancestors(paraphrase_id)

    assert source.object_id in ancestors
    assert read_id in ancestors
    assert summary_id in ancestors
