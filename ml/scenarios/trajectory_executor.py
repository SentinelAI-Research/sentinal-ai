from __future__ import annotations

from datetime import datetime, timezone

from backend.dataobjects.factory import create_data_object
from backend.dataobjects.models import DataObject
from backend.provenance.graph import ProvenanceGraph

from ml.scenarios.execution import ExecutedScenario
from ml.scenarios.models import ScenarioLabel


class ScenarioTrajectoryExecutor:
    """
    Converts an executed security scenario into a provenance-aware
    trajectory.

    Each transformation creates a new DataObject and a corresponding
    provenance edge.
    """

    @staticmethod
    def execute(
        scenario: ExecutedScenario,
        source_object: DataObject,
        source_text: str,
    ) -> tuple[ProvenanceGraph, ExecutedScenario]:

        graph = ProvenanceGraph()

        graph.add_object(source_object)

        current_object = source_object

        updated_steps = []

        for step in scenario.steps:
            step_timestamp = datetime.now(timezone.utc)

            output_object_id = f"{scenario.scenario_id}" f"-step-{step.step_number}"

            output_content = f"{step.tool_name}:" f"{source_text}"

            output_object = create_data_object(
                object_id=output_object_id,
                source=current_object.source,
                content=output_content,
                sensitivity=current_object.sensitivity,
                transformation=step.tool_name,
                parents=[current_object.object_id],
                created_step=(f"{scenario.scenario_id}" f"-step-{step.step_number}"),
            )

            graph.add_object(output_object)

            graph.add_relationship(
                current_object.object_id,
                output_object.object_id,
            )

            updated_steps.append(
                step.model_copy(
                    update={
                        "input_object_ids": [current_object.object_id],
                        "output_object_id": (output_object.object_id),
                        "timestamp": step_timestamp,
                    }
                )
            )

            current_object = output_object

        executed_trajectory = scenario.model_copy(
            update={
                "steps": updated_steps,
            }
        )

        return graph, executed_trajectory
