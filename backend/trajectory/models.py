from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class TrajectoryStep(BaseModel):
    """
    Represents one step in an agent's execution trajectory.

    For eg.
    Step: step_003
    Tool: summarize
    Input:  [file_001]
    Output: [summary_001]
    Verdict: ALLOW
    Risk: 0.12

    Diff:
    ToolCallProposal → what the agent wants to do.
    TrajectoryStep → what we recorded as part of the execution history.
    """

    step_id: str
    agent_id: str
    session_id: str
    
    tool_name: str
    arguments: dict

    input_data_objects: list[str] = Field(default_factory=list)
    output_data_objects: list[str] = Field(default_factory=list)

    verdict: Optional[str] = None
    risk_score: Optional[float] = None

    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )