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


def test_output_object_increases_transformation_depth():

    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    source_object = create_data_object(
        object_id="source",
        source="test-source",
        content="Original information.",
        sensitivity="PUBLIC",
        transformation=None,
        parents=[],
        created_step="source",
    )

    provenance_graph.add_object(source_object)

    policy_engine = PolicyEngine("policies/default.yaml")

    monitor = SentinelMonitor(
        policy_engine=policy_engine,
        risk_scorer=HeuristicRiskScorer(),
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
    )

    tool_executor = ToolExecutor()

    def transform(message: str):
        return f"Transformed: {message}"

    tool_executor.register_tool(
        "transform",
        transform,
    )

    runtime_executor = RuntimeExecutor(
        trajectory_store=trajectory_store,
        tool_executor=tool_executor,
    )

    output_handler = OutputDataObjectHandler(
        provenance_graph=provenance_graph,
    )

    service = MonitorService(
        monitor=monitor,
        runtime_executor=runtime_executor,
        output_handler=output_handler,
    )

    proposal = ToolCallProposal(
        step_id="step-1",
        agent_id="agent-1",
        tool_name="transform",
        arguments={
            "message": "Original information.",
        },
        input_data_objects=["source"],
    )

    decision, execution_result = service.process(
        proposal=proposal,
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"
    assert execution_result.executed is True

    assert provenance_graph.get_depth("step-1-output") == 1

    assert provenance_graph.get_transformation_depth("step-1-output") == 1
