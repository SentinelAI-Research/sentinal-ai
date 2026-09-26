from backend.monitor.executor import ToolExecutor
from backend.monitor.monitor import SentinelMonitor
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.service import MonitorService
from backend.monitor.tool_registry import create_tool_executor
from backend.policy.engine import PolicyEngine
from backend.provenance.graph import ProvenanceGraph
from backend.risk.heuristic import HeuristicRiskScorer
from backend.trajectory.store import TrajectoryStore


class SentinelRuntime:
    """
    Container for the complete SentinelAI runtime.

    This keeps the core security components together so that
    application code does not need to manually construct every
    component each time.
    """

    def __init__(
        self,
        monitor_service: MonitorService,
        trajectory_store: TrajectoryStore,
        provenance_graph: ProvenanceGraph,
        tool_executor: ToolExecutor,
    ):
        self.monitor_service = monitor_service
        self.trajectory_store = trajectory_store
        self.provenance_graph = provenance_graph
        self.tool_executor = tool_executor


def create_sentinel_runtime(
    policy_path: str = "policies/default.yaml",
) -> SentinelRuntime:
    """
    Create a fully configured SentinelAI runtime.
    """

    trajectory_store = TrajectoryStore()

    provenance_graph = ProvenanceGraph()

    policy_engine = PolicyEngine(policy_path)

    risk_scorer = HeuristicRiskScorer()

    tool_executor = create_tool_executor()

    runtime_executor = RuntimeExecutor(
        trajectory_store=trajectory_store,
        tool_executor=tool_executor,
    )

    monitor = SentinelMonitor(
        policy_engine=policy_engine,
        risk_scorer=risk_scorer,
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
    )

    output_handler = OutputDataObjectHandler(
        provenance_graph=provenance_graph,
    )

    monitor_service = MonitorService(
        monitor=monitor,
        runtime_executor=runtime_executor,
        output_handler=output_handler,
    )

    return SentinelRuntime(
        monitor_service=monitor_service,
        trajectory_store=trajectory_store,
        provenance_graph=provenance_graph,
        tool_executor=tool_executor,
    )
