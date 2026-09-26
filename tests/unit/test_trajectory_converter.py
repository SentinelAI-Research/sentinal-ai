from backend.policy.models import PolicyEvaluation
from backend.trajectory.models import TrajectoryStep
from ml.data.trajectory_converter import trajectory_to_training_record


def test_trajectory_converts_to_training_record():
    history = [
        TrajectoryStep(
            step_id="step-1",
            agent_id="agent-1",
            session_id="session-1",
            tool_name="read_file",
            arguments={"path": "example.txt"},
        ),
        TrajectoryStep(
            step_id="step-2",
            agent_id="agent-1",
            session_id="session-1",
            tool_name="summarize",
            arguments={},
        ),
    ]

    policy_result = PolicyEvaluation(
        verdict="WARN",
        matched_rules=["INTERNAL_TO_EXTERNAL"],
        reasons=["Internal data is being sent externally."],
    )

    record = trajectory_to_training_record(
        history=history,
        provenance_depth=2,
        destination="EXTERNAL",
        policy_result=policy_result,
        transformation_count=1,
        label="EARLY_RISK",
        scenario_id="summary-leak-001",
        scenario_family="summary_leakage",
        source_id="source-001",
    )

    assert record.trajectory_length == 2
    assert record.provenance_depth == 2
    assert record.external_destination == 1
    assert record.policy_warning == 1
    assert record.policy_block == 0
    assert record.transformation_count == 1
    assert record.label == "EARLY_RISK"
    assert record.scenario_id == "summary-leak-001"
    assert record.scenario_family == "summary_leakage"
    assert record.source_id == "source-001"