from backend.policy.models import PolicyEvaluation
from backend.trajectory.models import TrajectoryStep
from backend.risk.models import RiskAssessment


class HeuristicRiskScorer:
    """
    Transparent rule-based risk scorer.

    This is a baseline that will later be compared
    with the learned ML risk model.
    """

    def assess(
        self,
        policy_result: PolicyEvaluation,
        history: list[TrajectoryStep],
        provenance_depth: int,
        destination: str,
    ) -> RiskAssessment:

        risk_score = 0.0
        risk_factors: list[str] = []

        if policy_result.verdict == "BLOCK":
            risk_score = max(risk_score, 1.0)
            risk_factors.append("policy_block")

        elif policy_result.verdict == "WARN":
            risk_score = max(risk_score, 0.6)
            risk_factors.append("policy_warning")

        if provenance_depth >= 3:
            risk_score = max(risk_score, 0.5)
            risk_factors.append("deep_provenance")

        if len(history) >= 4:
            risk_score = max(risk_score, 0.4)
            risk_factors.append("long_trajectory")

        if destination == "EXTERNAL" and history:
            risk_score = max(risk_score, 0.5)
            risk_factors.append("external_destination")

        if not risk_factors:
            reason = "No heuristic risk factors detected."
        else:
            reason = (
                "Risk factors detected: "
                + ", ".join(risk_factors)
            )

        return RiskAssessment(
            risk_score=risk_score,
            risk_factors=risk_factors,
            reason=reason,
        )