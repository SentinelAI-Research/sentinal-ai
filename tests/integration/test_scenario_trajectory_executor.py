from backend.dataobjects.factory import create_data_object

from ml.scenarios.definitions import (
    long_horizon_leakage_scenario,
)
from ml.scenarios.execution import (
    SecurityScenarioExecutor,
)
from ml.scenarios.trajectory_executor import (
    ScenarioTrajectoryExecutor,
)


def test_long_horizon_scenario_creates_provenance_trajectory() -> None:
    scenario = long_horizon_leakage_scenario()

    executed = SecurityScenarioExecutor.execute(scenario)

    source_object = create_data_object(
        object_id="confidential-source-001",
        source="controlled-confidential-source",
        content=(
            "Controlled confidential information " "for SentinelAI trajectory testing."
        ),
        sensitivity="CONFIDENTIAL",
        transformation="source_ingestion",
        parents=[],
        created_step="scenario-source",
    )

    graph, trajectory = ScenarioTrajectoryExecutor.execute(
        scenario=executed,
        source_object=source_object,
        source_text=(
            "Controlled confidential information " "for SentinelAI trajectory testing."
        ),
    )

    assert len(trajectory.steps) == 6

    first_step = trajectory.steps[0]
    last_step = trajectory.steps[-1]

    assert first_step.input_object_ids == [source_object.object_id]

    assert first_step.output_object_id is not None

    assert last_step.output_object_id is not None

    assert graph.get_parents(first_step.output_object_id) == [source_object.object_id]

    assert graph.get_depth(last_step.output_object_id) == 6

    assert graph.get_transformation_depth(last_step.output_object_id) == 7

    assert last_step.tool_name == "send_message"
