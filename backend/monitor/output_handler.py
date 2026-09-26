from backend.dataobjects.factory import create_data_object
from backend.monitor.models import ExecutionResult
from backend.provenance.graph import ProvenanceGraph
from backend.schemas.contracts import ToolCallProposal

"""
Dedicated component responsible for converting a tool's output into a real DataObject
"""

class OutputDataObjectHandler:
    """
    Creates and registers DataObjects produced by allowed tool executions.

    The output object inherits the sensitivity of its input object.
    This is intentionally conservative: a transformation does not
    automatically make sensitive information less sensitive.
    """

    def __init__(self, provenance_graph: ProvenanceGraph):
        self.provenance_graph = provenance_graph

    def create_output_object(
        self,
        proposal: ToolCallProposal,
        execution_result: ExecutionResult,
        source_sensitivity: str,
    ) -> str:
        """
        Create a DataObject from a successful tool execution.

        Returns:
            The object_id of the newly created DataObject.
        """

        if not execution_result.executed:
            raise ValueError(
                "Cannot create an output DataObject for a tool "
                "execution that did not occur."
            )

        if execution_result.output is None:
            raise ValueError("Cannot create an output DataObject without tool output.")

        if not proposal.input_data_objects:
            raise ValueError(f"Proposal {proposal.step_id} has no input data objects.")

        output_object_id = f"{proposal.step_id}-output"

        output_object = create_data_object(
            object_id=output_object_id,
            source=f"tool:{proposal.tool_name}",
            content=execution_result.output,
            sensitivity=source_sensitivity,
            transformation=proposal.tool_name,
            parents=proposal.input_data_objects,
            created_step=proposal.step_id,
        )

        self.provenance_graph.add_object(output_object)

        for parent_id in proposal.input_data_objects:
            self.provenance_graph.add_relationship(
                parent_id,
                output_object_id,
            )

        return output_object_id
