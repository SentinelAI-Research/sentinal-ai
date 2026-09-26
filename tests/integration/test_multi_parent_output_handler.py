from backend.dataobjects.factory import create_data_object
from backend.monitor.models import ExecutionResult
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.provenance.graph import ProvenanceGraph
from backend.schemas.contracts import ToolCallProposal


def test_output_handler_preserves_multiple_input_parents():
    graph = ProvenanceGraph()
    handler = OutputDataObjectHandler(provenance_graph=graph)

    source_a = create_data_object(
        object_id="source-a",
        source="dataset/source_a.txt",
        content="Information from source A.",
        sensitivity="INTERNAL",
        transformation=None,
        parents=[],
        created_step="source-a",
    )

    source_b = create_data_object(
        object_id="source-b",
        source="dataset/source_b.txt",
        content="Information from source B.",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source-b",
    )

    graph.add_object(source_a)
    graph.add_object(source_b)

    proposal = ToolCallProposal(
        step_id="step-multi-parent",
        agent_id="test-agent",
        tool_name="summarize",
        arguments={"text": "combined information"},
        input_data_objects=[
            "source-a",
            "source-b",
        ],
    )

    execution_result = ExecutionResult(
        executed=True,
        tool_name="summarize",
        output="Combined summary",
    )

    output_id = handler.create_output_object(
        proposal=proposal,
        execution_result=execution_result,
        source_sensitivity="CONFIDENTIAL",
    )

    output_object = graph.get_object(output_id)

    assert output_id == "step-multi-parent-output"

    assert output_object.parents == [
        "source-a",
        "source-b",
    ]

    assert "source-a" in graph.get_parents(output_id)
    assert "source-b" in graph.get_parents(output_id)

    assert graph.get_depth(output_id) == 1
