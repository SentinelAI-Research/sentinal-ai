from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from ml.scenarios.models import ScenarioLabel


class SecurityTrainingRecord(BaseModel):
    """
    One step-level supervised training example generated
    from an executed security trajectory.
    """

    trajectory_id: str
    scenario_id: str
    scenario_type: str

    step_number: int = Field(ge=1)
    total_steps: int = Field(ge=1)

    tool_name: str

    input_object_ids: list[str] = Field(default_factory=list)
    output_object_id: str | None = None

    label: ScenarioLabel

    timestamp: datetime

    provenance_depth: int = Field(ge=0)
    transformation_depth: int = Field(ge=0)