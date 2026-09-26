from backend.agent.proposal_builder import ProposalBuilder
from backend.dataobjects.registry import DataObjectRegistry
from backend.monitor.factory import create_sentinel_runtime


def test_runtime_executes_allowed_calculation():
    runtime = create_sentinel_runtime()

    registry = DataObjectRegistry(provenance_graph=runtime.provenance_graph)

    source = registry.register_source(
        object_id="public-source",
        source="benchmark/public",
        content="2 + 3",
        sensitivity="PUBLIC",
        created_step="source",
    )

    builder = ProposalBuilder(agent_id="allow-test-agent")

    proposal = builder.build(
        step_id="allow-calculation",
        tool_name="calculate",
        arguments={
            "expression": "2 + 3",
        },
        input_data_objects=[source.object_id],
    )

    decision, execution_result = runtime.monitor_service.process(
        proposal=proposal,
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"
    assert execution_result.executed is True
    assert execution_result.error is None
    assert execution_result.output is not None
    assert execution_result.output_data_object_id is not None

    output_object = runtime.provenance_graph.get_object(
        execution_result.output_data_object_id
    )

    assert output_object.parents == [source.object_id]
