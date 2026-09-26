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
    ):
        # ---------------------------------------------------------
        # 1. Evaluate the proposed action
        # ---------------------------------------------------------

        decision = self.monitor.evaluate(
            proposal=proposal,
            destination=destination,
        )

        # ---------------------------------------------------------
        # 2. Enforce the security decision
        # ---------------------------------------------------------

        execution_result = self.runtime_executor.apply_decision(
            step_id=proposal.step_id,
            decision=decision,
            tool_name=proposal.tool_name,
            arguments=proposal.arguments,
        )

        # ---------------------------------------------------------
        # 3. Create a provenance object only if execution succeeded
        # ---------------------------------------------------------

        if execution_result.executed:
            input_object_id = proposal.input_data_objects[-1]

            source_object = self.monitor.provenance_graph.get_object(input_object_id)

            output_object_id = self.output_handler.create_output_object(
                proposal=proposal,
                execution_result=execution_result,
                source_sensitivity=source_object.sensitivity,
            )

            execution_result.output_data_object_id = output_object_id

        return decision, execution_result
