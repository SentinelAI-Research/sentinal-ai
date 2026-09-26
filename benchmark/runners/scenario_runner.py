from dataclasses import dataclass

from backend.trajectory.models import TrajectoryStep
from backend.trajectory.store import TrajectoryStore
from benchmark.scenarios.models import SecurityScenario

"""
Proper component that records scenario execution into our existing TrajectoryStore.
"""
@dataclass
class ScenarioExecution:
    scenario: SecurityScenario
    trajectory: list[TrajectoryStep]


class ScenarioRunner:
    def __init__(self, trajectory_store: TrajectoryStore):
        self.trajectory_store = trajectory_store

    def record_step(
        self,
        scenario: SecurityScenario,
        step_id: str,
        agent_id: str,
        session_id: str,
        tool_name: str,
        arguments: dict,
        input_data_objects: list[str] | None = None,
        output_data_objects: list[str] | None = None,
    ) -> TrajectoryStep:

        step = TrajectoryStep(
            step_id=step_id,
            agent_id=agent_id,
            session_id=session_id,
            tool_name=tool_name,
            arguments=arguments,
            input_data_objects=input_data_objects or [],
            output_data_objects=output_data_objects or [],
        )

        self.trajectory_store.add_step(step)

        return step

    def get_execution(
        self,
        scenario: SecurityScenario,
        session_id: str,
    ) -> ScenarioExecution:

        trajectory = self.trajectory_store.get_session_steps(session_id)

        return ScenarioExecution(
            scenario=scenario,
            trajectory=trajectory,
        )