from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel, Field


class ToolCallProposal(BaseModel):
    """
    Represents a tool action proposed by the agent.

    SentinelAI receives this proposal before the tool is executed.
    """

    step_id: str
    agent_id: str

    tool_name: str
    arguments: dict

    input_data_objects: list[str] = Field(default_factory=list)

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


class Decision(BaseModel):
    """
    Represents SentinelAI's decision for a proposed tool call.
    """

    step_id: str

    verdict: str
    risk_score: float
    risk_probability: Optional[float] = None

    policy_hits: list[str] = Field(default_factory=list)

    provenance_trace: list[str] = Field(default_factory=list)

    reason: str