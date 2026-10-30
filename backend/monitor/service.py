from collections.abc import Callable
from typing import Any

from backend.monitor.monitor import SentinelMonitor
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.schemas.contracts import ToolCallProposal


class MonitorService:
    """
    Coordinates security evaluation, runtime enforcement,
    and provenance creation for successful tool executions.
    """

    def __init__(
        self,
        monitor: SentinelMonitor,
        runtime_executor: RuntimeExecutor,
        output_handler: OutputDataObjectHandler,
    ):
        self.monitor = monitor
        self.runtime_executor = runtime_executor
        self.output_handler = output_handler

    def process(
        self,
        proposal: ToolCallProposal,
        destination: str,
        event_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ):
        # ---------------------------------------------------------
        # 1. Evaluate the proposed action
        # ---------------------------------------------------------

        decision = self.monitor.evaluate(
            proposal=proposal,
            destination=destination,
        )
        if event_callback:
            event_callback(
                "monitor_decision",
                {
                    "step_id": proposal.step_id,
                    "verdict": decision.verdict,
                    "reason": decision.reason,
                    "matched_policy_rules": decision.policy_hits,
                    "risk_score": decision.risk_score,
                    "risk_probability": decision.risk_probability,
                    "provenance_reference_ids": decision.provenance_trace,
                },
            )

        # ---------------------------------------------------------
        # 2. Enforce the security decision
        # ---------------------------------------------------------

        if event_callback and decision.verdict != "ALLOW":
            event_callback(
                "execution_skipped",
                {
                    "step_id": proposal.step_id,
                    "reason": "Monitor verdict was not ALLOW.",
                },
            )
        elif event_callback:
            event_callback(
                "execution_attempt",
                {"step_id": proposal.step_id, "tool_name": proposal.tool_name},
            )

        execution_result = self.runtime_executor.apply_decision(
            step_id=proposal.step_id,
            decision=decision,
            tool_name=proposal.tool_name,
            arguments=proposal.arguments,
        )
        if event_callback:
            event_callback(
                "execution_result",
                {
                    "step_id": proposal.step_id,
                    "verdict": decision.verdict,
                    "executed": execution_result.executed,
                    "error": execution_result.error,
                },
            )

        # ---------------------------------------------------------
        # 3. Create a provenance object only if execution succeeded
        # ---------------------------------------------------------

        if execution_result.executed:
            try:
                output_object_id = self.output_handler.create_output_object(
                    proposal=proposal,
                    execution_result=execution_result,
                )
            except Exception as exc:
                execution_result.error = (
                    "Tool executed, but output provenance registration failed: "
                    f"{exc}"
                )
                if event_callback:
                    event_callback(
                        "output_provenance_failed",
                        {
                            "step_id": proposal.step_id,
                            "error": execution_result.error,
                        },
                    )
            else:
                execution_result.output_data_object_id = output_object_id
                if event_callback:
                    event_callback(
                        "output_provenance_recorded",
                        {
                            "step_id": proposal.step_id,
                            "output_data_object_id": output_object_id,
                        },
                    )

        return decision, execution_result
