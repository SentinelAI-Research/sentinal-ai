from backend.dataobjects.factory import create_data_object
from backend.monitor.executor import ToolExecutor
from backend.monitor.monitor import SentinelMonitor
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.service import MonitorService
from backend.policy.engine import PolicyEngine
from backend.provenance.graph import ProvenanceGraph
from backend.risk.heuristic import HeuristicRiskScorer
from backend.schemas.contracts import ToolCallProposal
from backend.trajectory.store import TrajectoryStore


def test_output_preserves_multiple_parents():

    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    source_a = create_data_object(
        object_id="source-a",
        source="document-a",
        content="Information A",
        sensitivity="PUBLIC",
        transformation=None,
        parents=[],
        created_step="source-a",
    )

    source_b = create_data_object(
        object_id="source-b",
        source="document-b",
        content="Information B",
        sensitivity="PUBLIC",
        transformation=None,
        parents=[],
        created_step="source-b",
    )

    provenance_graph.add_object(source_a)
    provenance_graph.add_object(source_b)

    output_handler = OutputDataObjectHandler(
        provenance_graph=provenance_graph,
    )

    tool_executor = ToolExecutor()

    def combine(message_a: str, message_b: str):
        return f"{message_a} + {message_b}"

    tool_executor.register_tool(
        "combine",
        combine,
    )

    runtime_executor = RuntimeExecutor(
        trajectory_store=trajectory_store,
        tool_executor=tool_executor,
    )

    monitor = SentinelMonitor(
        policy_engine=PolicyEngine("policies/default.yaml"),
        risk_scorer=HeuristicRiskScorer(),
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
    )

    service = MonitorService(
        monitor=monitor,
        runtime_executor=runtime_executor,
        output_handler=output_handler,
    )

    proposal = ToolCallProposal(
        step_id="combine-step",
        agent_id="agent-1",
        tool_name="combine",
        arguments={
            "message_a": "Information A",
            "message_b": "Information B",
        },
        input_data_objects=[
            "source-a",
            "source-b",
        ],
    )

    decision, execution_result = service.process(
        proposal=proposal,
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"
    assert execution_result.executed is True

    output_object = provenance_graph.get_object("combine-step-output")

    assert output_object.parents == [
        "source-a",
        "source-b",
    ]

    assert set(provenance_graph.get_parents("combine-step-output")) == {
        "source-a",
        "source-b",
    }
