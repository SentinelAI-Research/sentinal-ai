from backend.monitor.monitor import SentinelMonitor
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.executor import ToolExecutor
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.policy.engine import PolicyEngine
from backend.risk.heuristic import HeuristicRiskScorer
from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.store import TrajectoryStore
from backend.schemas.contracts import ToolCallProposal
from backend.dataobjects.factory import create_data_object
from backend.monitor.service import MonitorService


def test_confidential_external_action_is_blocked():

    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    source_object = create_data_object(
        object_id="test-confidential-source",
        source="test-source",
        content="This is confidential project information.",
        sensitivity="CONFIDENTIAL",
        transformation=None,
        parents=[],
        created_step="source",
    )

    provenance_graph.add_object(source_object)

    policy_engine = PolicyEngine("policies/default.yaml")

    risk_scorer = HeuristicRiskScorer()

    monitor = SentinelMonitor(
        policy_engine=policy_engine,
        risk_scorer=risk_scorer,
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
    )

    tool_executor = ToolExecutor()

    def should_not_run(recipient: str, message: str):
        raise AssertionError("Blocked tool should never execute.")

    tool_executor.register_tool(
        "send_message",
        should_not_run,
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
        agent_id="test-agent",
        tool_name="send_message",
        arguments={
            "recipient": "external@example.com",
            "message": "Confidential project information",
        },
        input_data_objects=["test-confidential-source"],
    )

    decision, execution_result = service.process(
        proposal=proposal,
        destination="EXTERNAL",
    )

    assert decision.verdict == "BLOCK"
    assert decision.risk_score == 1.0
    assert execution_result.executed is False

    assert execution_result.output_data_object_id is None

    assert "step-1-output" not in provenance_graph.graph
