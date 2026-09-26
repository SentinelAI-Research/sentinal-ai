from backend.policy.models import PolicyEvaluation
from backend.risk.heuristic import HeuristicRiskScorer


def test_policy_block_produces_maximum_risk():
    scorer = HeuristicRiskScorer()

    policy_result = PolicyEvaluation(
        verdict="BLOCK",
        matched_rules=["CONFIDENTIAL_TO_EXTERNAL"],
    )

    result = scorer.assess(
        policy_result=policy_result,
        history=[],
        provenance_depth=0,
        destination="EXTERNAL",
    )

    assert result.risk_score == 1.0
    assert "policy_block" in result.reason


def test_policy_warning_produces_warning_risk():
    scorer = HeuristicRiskScorer()

    policy_result = PolicyEvaluation(
        verdict="WARN",
        matched_rules=["INTERNAL_TO_EXTERNAL"],
    )

    result = scorer.assess(
        policy_result=policy_result,
        history=[],
        provenance_depth=0,
        destination="EXTERNAL",
    )

    assert result.risk_score == 0.6
    assert "policy_warning" in result.reason


def test_deep_provenance_increases_risk():
    scorer = HeuristicRiskScorer()

    policy_result = PolicyEvaluation(
        verdict="ALLOW",
        matched_rules=[],
    )

    result = scorer.assess(
        policy_result=policy_result,
        history=[],
        provenance_depth=3,
        destination="INTERNAL",
    )

    assert result.risk_score == 0.5
    assert "deep_provenance" in result.reason
