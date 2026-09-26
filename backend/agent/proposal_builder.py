from backend.schemas.contracts import ToolCallProposal


class ProposalBuilder:
    """
    Creates ToolCallProposal objects for the agent runtime.

    Session identity is maintained outside ToolCallProposal because
    the current contract does not contain a session_id field.
    """

    def __init__(
        self,
        agent_id: str,
    ):
        self.agent_id = agent_id

    def build(
        self,
        step_id: str,
        tool_name: str,
        arguments: dict,
        input_data_objects: list[str],
    ) -> ToolCallProposal:
        """
        Build a tool-call proposal.
        """

        if not step_id:
            raise ValueError("step_id cannot be empty.")

        if not tool_name:
            raise ValueError("tool_name cannot be empty.")

        if not input_data_objects:
            raise ValueError("At least one input DataObject is required.")

        return ToolCallProposal(
            step_id=step_id,
            agent_id=self.agent_id,
            tool_name=tool_name,
            arguments=arguments,
            input_data_objects=input_data_objects,
        )
