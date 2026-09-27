from __future__ import annotations

from backend.provenance.graph import ProvenanceGraph
from ml.scenarios.execution import ExecutedScenario
from ml.scenarios.labeled_records import SecurityTrainingRecord


class SecurityLabelBuilder:
    """
    Converts an executed security scenario into step-level
    supervised training records.

    Each tool-call step becomes one training example.
    """

    @staticmethod
    def build(
        scenario: ExecutedScenario,
        graph: ProvenanceGraph,
    ) -> list[SecurityTrainingRecord]:

        records: list[SecurityTrainingRecord] = []

        total_steps = len(scenario.steps)

        if total_steps == 0:
            raise ValueError(f"Scenario '{scenario.scenario_id}' contains no steps.")

        for step in scenario.steps:
            if not step.output_object_id:
                raise ValueError(
                    f"Step {step.step_number} in scenario "
                    f"'{scenario.scenario_id}' has no output object ID."
                )

            if step.output_object_id not in graph.graph:
                raise ValueError(
                    f"Output object '{step.output_object_id}' "
                    f"for step {step.step_number} is not present "
                    f"in the provenance graph."
                )

            provenance_depth = graph.get_depth(step.output_object_id)

            transformation_depth = graph.get_transformation_depth(step.output_object_id)

            record = SecurityTrainingRecord(
                trajectory_id=scenario.scenario_id,
                scenario_id=scenario.scenario_id,
                scenario_type=scenario.scenario_type.value,
                step_number=step.step_number,
                total_steps=total_steps,
                tool_name=step.tool_name,
                input_object_ids=step.input_object_ids,
                output_object_id=step.output_object_id,
                label=step.label,
                timestamp=step.timestamp,
                provenance_depth=provenance_depth,
                transformation_depth=transformation_depth,
            )

            records.append(record)

        return records
