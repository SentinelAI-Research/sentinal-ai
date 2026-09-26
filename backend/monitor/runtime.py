from backend.monitor.executor import ToolExecutor
from backend.monitor.models import ExecutionResult
from backend.schemas.contracts import Decision
from backend.trajectory.store import TrajectoryStore


class RuntimeExecutor:
    def __init__(
        self,
        trajectory_store: TrajectoryStore,
        tool_executor: ToolExecutor,
    ):
        self.trajectory_store = trajectory_store
        self.tool_executor = tool_executor

    def apply_decision(
        self,
        step_id: str,
        decision: Decision,
        tool_name: str,
        arguments: dict,
    ) -> ExecutionResult:

        self.trajectory_store.update_decision(
            step_id=step_id,
            verdict=decision.verdict,
            risk_score=decision.risk_score,
        )

        if decision.verdict != "ALLOW":
            return ExecutionResult(
                executed=False,
                tool_name=tool_name,
                error=(f"Tool execution prevented by " f"{decision.verdict} decision."),
            )

        try:
            output = self.tool_executor.execute(
                tool_name,
                arguments,
            )

            return ExecutionResult(
                executed=True,
                tool_name=tool_name,
                output=str(output),
            )

        except Exception as exc:
            return ExecutionResult(
                executed=False,
                tool_name=tool_name,
                error=str(exc),
            )
