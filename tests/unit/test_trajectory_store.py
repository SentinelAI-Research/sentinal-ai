from backend.trajectory.models import TrajectoryStep
from backend.trajectory.store import TrajectoryStore


def test_update_decision():
    store = TrajectoryStore()

    step = TrajectoryStep(
        step_id="step-1",
        agent_id="agent-1",
        session_id="session-1",
        tool_name="send_message",
        arguments={},
    )

    store.add_step(step)

    store.update_decision(
        step_id="step-1",
        verdict="BLOCK",
        risk_score=1.0,
    )

    updated = store.get_step("step-1")

    assert updated.verdict == "BLOCK"
    assert updated.risk_score == 1.0