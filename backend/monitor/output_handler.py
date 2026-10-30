from backend.dataobjects.factory import create_data_object
from backend.dataobjects.sensitivity import most_sensitive_sensitivity
from backend.monitor.models import ExecutionResult
from backend.provenance.graph import ProvenanceGraph
from backend.schemas.contracts import ToolCallProposal

"""
Dedicated component responsible for converting a tool's output into a real DataObject
"""


class OutputDataObjectHandler:
    """
    Creates and registers DataObjects produced by allowed tool executions.

    The output inherits the highest sensitivity among all parents.
    Supported runtime labels follow PUBLIC < INTERNAL < CONFIDENTIAL.
    Unknown labels fail closed.
    """

    def __init__(self, provenance_graph: ProvenanceGraph):
        self.provenance_graph = provenance_graph

    def create_output_object(
        self,
        proposal: ToolCallProposal,
        execution_result: ExecutionResult,
        source_sensitivity: str | None = None,
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

        parent_objects = self.provenance_graph.resolve_objects(
            proposal.input_data_objects
        )
        sensitivities = [parent.sensitivity for parent in parent_objects]
        if source_sensitivity is not None:
            # Keep the former argument as an additional conservative constraint.
            sensitivities.append(source_sensitivity)
        output_sensitivity = most_sensitive_sensitivity(sensitivities)

        output_object_id = f"{proposal.step_id}-output"
        if self.provenance_graph.has_object(output_object_id):
            raise ValueError(f"Data object already exists: {output_object_id}")

        output_object = create_data_object(
            object_id=output_object_id,
            source=f"tool:{proposal.tool_name}",
            content=execution_result.output,
            sensitivity=output_sensitivity,
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
