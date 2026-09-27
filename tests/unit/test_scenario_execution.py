from ml.scenarios.definitions import (
    long_horizon_leakage_scenario,
)
from ml.scenarios.execution import (
    SecurityScenarioExecutor,
)
from ml.scenarios.models import ScenarioLabel


def test_long_horizon_scenario_execution() -> None:
    scenario = long_horizon_leakage_scenario()

    executed = SecurityScenarioExecutor.execute(scenario)

    assert executed.scenario_id == "long-horizon-leakage-001"

    assert len(executed.steps) == 6

    assert [step.step_number for step in executed.steps] == [1, 2, 3, 4, 5, 6]

    assert [step.tool_name for step in executed.steps] == [
        "read_file",
        "summarize",
        "write_memory",
        "read_memory",
        "paraphrase",
        "send_message",
    ]

    assert executed.steps[-1].label == ScenarioLabel.VIOLATION

    assert executed.final_label == ScenarioLabel.VIOLATION

    for step in executed.steps:
        assert step.timestamp is not None
