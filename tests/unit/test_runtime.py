from backend.schemas.contracts import Decision
from backend.trajectory.models import TrajectoryStep
from backend.trajectory.store import TrajectoryStore
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.executor import ToolExecutor


def test_block_decision_prevents_execution():
    store = TrajectoryStore()

    step = TrajectoryStep(
        step_id="step-1",
        agent_id="agent-1",
        session_id="session-1",
        tool_name="send_message",
        arguments={
            "recipient": "external@example.com",
        },
        input_data_objects=["object-1"],
    )

    store.add_step(step)

    tool_executor = ToolExecutor()

    executor = RuntimeExecutor(
        trajectory_store=store,
        tool_executor=tool_executor,
    )

    decision = Decision(
        step_id="step-1",
        verdict="BLOCK",
        risk_score=1.0,
        risk_probability=None,
        policy_hits=["CONFIDENTIAL_TO_EXTERNAL"],
        provenance_trace=["object-1"],
        reason="Confidential data cannot be sent externally.",
    )

    result = executor.apply_decision(
        step_id="step-1",
        decision=decision,
        tool_name="send_message",
        arguments={
            "recipient": "external@example.com",
            "message": "secret",
        },
    )

    assert result.executed is False

    updated = store.get_step("step-1")

    assert updated.verdict == "BLOCK"
    assert updated.risk_score == 1.0


def test_allow_decision_permits_execution():
    store = TrajectoryStore()

    step = TrajectoryStep(
        step_id="step-2",
        agent_id="agent-1",
        session_id="session-1",
        tool_name="echo",
        arguments={
            "message": "hello SentinelAI",
        },
    )

    store.add_step(step)

    tool_executor = ToolExecutor()

    def echo(message: str):
        return message

    tool_executor.register_tool(
        "echo",
        echo,
    )

    executor = RuntimeExecutor(
        trajectory_store=store,
        tool_executor=tool_executor,
    )

    decision = Decision(
        step_id="step-2",
        verdict="ALLOW",
        risk_score=0.0,
        risk_probability=None,
        policy_hits=[],
        provenance_trace=[],
        reason="No security policy violation detected.",
    )

    result = executor.apply_decision(
        step_id="step-2",
        decision=decision,
        tool_name="echo",
        arguments={
            "message": "hello SentinelAI",
        },
    )

    assert result.executed is True
    assert result.output == "hello SentinelAI"

    updated = store.get_step("step-2")

    assert updated.verdict == "ALLOW"
    assert updated.risk_score == 0.0