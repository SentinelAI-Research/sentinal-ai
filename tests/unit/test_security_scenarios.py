from ml.scenarios.definitions import all_scenarios
from ml.scenarios.models import (
    ScenarioLabel,
    ScenarioType,
)


def test_all_security_scenarios_exist() -> None:
    scenarios = all_scenarios()

    assert len(scenarios) == 4

    scenario_types = {scenario.scenario_type for scenario in scenarios}

    assert ScenarioType.BENIGN_INTERNAL in scenario_types
    assert ScenarioType.CONFIDENTIAL_MEMORY in scenario_types
    assert ScenarioType.CONFIDENTIAL_EXTERNAL in scenario_types
    assert ScenarioType.LONG_HORIZON_LEAKAGE in scenario_types


def test_scenarios_have_valid_labels() -> None:
    scenarios = all_scenarios()

    for scenario in scenarios:
        assert scenario.steps
        assert scenario.final_label in {
            ScenarioLabel.SAFE,
            ScenarioLabel.VIOLATION,
        }

        step_numbers = [step.step_number for step in scenario.steps]

        assert step_numbers == list(range(1, len(step_numbers) + 1))


def test_long_horizon_scenario_has_multiple_steps() -> None:
    scenarios = all_scenarios()

    scenario = next(
        scenario
        for scenario in scenarios
        if scenario.scenario_type == ScenarioType.LONG_HORIZON_LEAKAGE
    )

    assert len(scenario.steps) == 6

    assert scenario.steps[-1].label == ScenarioLabel.VIOLATION
