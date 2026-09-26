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


def test_blocked_action_does_not_create_provenance_output():

    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    confidential_object = create_data_object(
        object_id="confidential-source",
        source="confidential-document",
        content="Private information.",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source",
    )

    provenance_graph.add_object(confidential_object)

    tool_executor = ToolExecutor()

    def forbidden_action(message: str):
        raise AssertionError("Forbidden action was executed.")

    tool_executor.register_tool(
        "forbidden_action",
        forbidden_action,
    )

    monitor = SentinelMonitor(
        policy_engine=PolicyEngine("policies/default.yaml"),
        risk_scorer=HeuristicRiskScorer(),
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
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
        step_id="blocked-step",
        agent_id="agent-1",
        tool_name="forbidden_action",
        arguments={"message": "Private information."},
        input_data_objects=["confidential-source"],
    )

    decision, execution_result = service.process(
        proposal=proposal,
        destination="EXTERNAL",
    )

    assert decision.verdict == "BLOCK"
    assert execution_result.executed is False

    assert execution_result.output_data_object_id is None

    assert "blocked-step-output" not in provenance_graph.graph
