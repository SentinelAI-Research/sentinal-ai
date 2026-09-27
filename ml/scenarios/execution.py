from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ml.scenarios.models import (
    ScenarioLabel,
    SecurityScenario,
)


class ExecutedScenarioStep(BaseModel):
    step_number: int = Field(ge=1)
    tool_name: str
    input_object_ids: list[str] = Field(default_factory=list)
    output_object_id: str | None = None
    label: ScenarioLabel
    timestamp: datetime


class ExecutedScenario(BaseModel):
    scenario_id: str
    scenario_type: str
    steps: list[ExecutedScenarioStep] = Field(default_factory=list)
    final_label: ScenarioLabel


class SecurityScenarioExecutor:
    """
    Executes a security scenario definition and records
    the resulting trajectory.

    At this stage the executor focuses on creating a
    deterministic trajectory record. Actual tool execution
    and provenance-linked DataObjects will be connected
    in the next step.
    """

    @staticmethod
    def execute(
        scenario: SecurityScenario,
    ) -> ExecutedScenario:
        executed_steps: list[ExecutedScenarioStep] = []

        for step in scenario.steps:
            executed_steps.append(
                ExecutedScenarioStep(
                    step_number=step.step_number,
                    tool_name=step.tool_name,
                    input_object_ids=list(step.input_object_ids),
                    output_object_id=step.output_object_id,
                    label=step.label,
                    timestamp=datetime.now(timezone.utc),
                )
            )

        return ExecutedScenario(
            scenario_id=scenario.scenario_id,
            scenario_type=scenario.scenario_type.value,
            steps=executed_steps,
            final_label=scenario.final_label,
        )
