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


def test_allowed_execution_creates_output_provenance():

    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()

    source_object = create_data_object(
        object_id="public-source",
        source="test-source",
        content="Public information for testing.",
        sensitivity="PUBLIC",
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

    def echo(message: str):
        return message

    tool_executor.register_tool(
        "echo",
        echo,
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
        tool_name="echo",
        arguments={
            "message": "Hello SentinelAI",
        },
        input_data_objects=["public-source"],
    )

    decision, execution_result = service.process(
        proposal=proposal,
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"
    assert execution_result.executed is True
    assert execution_result.output == "Hello SentinelAI"

    assert execution_result.output_data_object_id == "step-1-output"

    output_object = provenance_graph.get_object("step-1-output")

    assert output_object.source == "tool:echo"
    assert output_object.transformation == "echo"
    assert output_object.sensitivity == "PUBLIC"

    assert output_object.parents == ["public-source"]

    assert provenance_graph.get_parents("step-1-output") == ["public-source"]
