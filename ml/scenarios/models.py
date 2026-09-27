from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ScenarioLabel(str, Enum):
    SAFE = "SAFE"
    EARLY_RISK = "EARLY_RISK"
    VIOLATION = "VIOLATION"


class ScenarioType(str, Enum):
    BENIGN_INTERNAL = "benign_internal"
    CONFIDENTIAL_MEMORY = "confidential_memory"
    CONFIDENTIAL_EXTERNAL = "confidential_external"
    LONG_HORIZON_LEAKAGE = "long_horizon_leakage"


class ScenarioStep(BaseModel):
    step_number: int = Field(ge=1)
    tool_name: str
    input_object_ids: list[str] = Field(default_factory=list)
    output_object_id: str | None = None
    label: ScenarioLabel


class SecurityScenario(BaseModel):
    scenario_id: str
    scenario_type: ScenarioType
    description: str
    steps: list[ScenarioStep]
    final_label: ScenarioLabel
