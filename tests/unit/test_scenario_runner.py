from backend.trajectory.store import TrajectoryStore
from benchmark.runners.scenario_runner import ScenarioRunner
from benchmark.scenarios.direct_leakage import (
    create_direct_leakage_scenario,
)

def test_scenario_runner_records_trajectory():
    store = TrajectoryStore()
    runner = ScenarioRunner(store)

    scenario = create_direct_leakage_scenario()

    runner.record_step(
        scenario=scenario,
        step_id="step-1",
        agent_id="agent-1",
        session_id="session-1",
        tool_name="read_file",
        arguments={"path": "confidential.txt"},
    )

    runner.record_step(
        scenario=scenario,
        step_id="step-2",
        agent_id="agent-1",
        session_id="session-1",
        tool_name="send_message",
        arguments={"destination": "EXTERNAL"},
        input_data_objects=["source-1"],
    )

    execution = runner.get_execution(
        scenario,
        "session-1",
    )

    assert len(execution.trajectory) == 2
    assert execution.trajectory[0].tool_name == "read_file"
    assert execution.trajectory[1].tool_name == "send_message"