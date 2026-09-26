from backend.policy.models import PolicyEvaluation
from backend.risk.heuristic import HeuristicRiskScorer
from backend.trajectory.models import TrajectoryStep


def test_block_policy_produces_high_risk():
    scorer = HeuristicRiskScorer()

    result = scorer.assess(
        policy_result=PolicyEvaluation(
            verdict="BLOCK",
            matched_rules=["CONFIDENTIAL_TO_EXTERNAL"],
            reasons=["Confidential data must not be sent externally."],
        ),
        history=[],
        provenance_depth=0,
        destination="EXTERNAL",
    )

    assert result.risk_score == 1.0
    assert "policy_block" in result.risk_factors


def test_long_trajectory_adds_risk_factor():
    scorer = HeuristicRiskScorer()

    history = [
        TrajectoryStep(
            step_id=f"step_{i}",
            agent_id="agent_001",
            session_id="session_A",
            tool_name="read_file",
            arguments={},
        )
        for i in range(4)
    ]

    result = scorer.assess(
        policy_result=PolicyEvaluation(verdict="ALLOW"),
        history=history,
        provenance_depth=4,
        destination="EXTERNAL",
    )

    assert "long_trajectory" in result.risk_factors
    assert "deep_provenance" in result.risk_factors
    assert "external_destination" in result.risk_factors