from pathlib import Path

from backend.monitor.config import MonitorConfig
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
    """

    def __init__(
        self,
        monitor_service: MonitorService,
        trajectory_store: TrajectoryStore,
        provenance_graph: ProvenanceGraph,
        tool_executor: ToolExecutor,
        monitor: SentinelMonitor | None = None,
        runtime_executor: RuntimeExecutor | None = None,
        output_handler: OutputDataObjectHandler | None = None,
        config: MonitorConfig | None = None,
    ):
        self.monitor_service = monitor_service
        self.trajectory_store = trajectory_store
        self.provenance_graph = provenance_graph
        self.tool_executor = tool_executor
        self.monitor = monitor
        self.runtime_executor = runtime_executor
        self.output_handler = output_handler
        self.config = config


def create_sentinel_runtime(
    policy_path: str = "policies/default.yaml",
    file_root: str | Path | None = None,
    memory_path: str | Path | None = None,
    monitor_config: MonitorConfig | None = None,
) -> SentinelRuntime:
    """
    Create a fully configured SentinelAI runtime under the given MonitorConfig.
    """
    cfg = monitor_config or MonitorConfig.from_preset("C4")
    trajectory_store = TrajectoryStore()
    provenance_graph = ProvenanceGraph()
    policy_engine = PolicyEngine(policy_path)
    risk_scorer = HeuristicRiskScorer()

    tool_executor = create_tool_executor(
        file_root=file_root,
        memory_path=memory_path,
    )

    runtime_executor = RuntimeExecutor(
        trajectory_store=trajectory_store,
        tool_executor=tool_executor,
    )

    monitor = SentinelMonitor(
        policy_engine=policy_engine,
        risk_scorer=risk_scorer,
        provenance_graph=provenance_graph,
        trajectory_store=trajectory_store,
        config=cfg,
    )

    output_handler = OutputDataObjectHandler(
        provenance_graph=provenance_graph,
        inherit_sensitivity=cfg.inherit_sensitivity,
        memory_preserves_sensitivity=cfg.memory_preserves_sensitivity,
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
        monitor=monitor,
        runtime_executor=runtime_executor,
        output_handler=output_handler,
        config=cfg,
    )
