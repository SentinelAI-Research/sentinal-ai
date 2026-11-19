from backend.policy.models import PolicyEvaluation
from backend.risk.models import RiskAssessment
from backend.trajectory.models import TrajectoryStep


class HeuristicRiskScorer:
    """
    Transparent rule-based risk scorer with early-warning transformation
    depth escalation (B4.2) and ablation toggles (B5 / R5).
    """

    def assess(
        self,
        policy_result: PolicyEvaluation,
        history: list[TrajectoryStep],
        provenance_depth: int,
        destination: str,
        *,
        tool_name: str = "",
        has_sensitive_ancestor: bool = False,
        sensitive_ancestor_count: int = 0,
        transformation_depth: int = 0,
        use_provenance_features: bool = True,
        use_transformation_depth_signal: bool = True,
        use_trajectory_features: bool = True,
    ) -> RiskAssessment:
        risk_score = 0.0
        risk_factors: list[str] = []

        if policy_result.verdict == "BLOCK":
            risk_score = max(risk_score, 1.0)
            risk_factors.append("policy_block")
        elif policy_result.verdict == "WARN" and destination != "MEMORY":
            risk_score = max(risk_score, 0.50)
            risk_factors.append("policy_warning")

        if use_trajectory_features:
            if len(history) >= 4:
                risk_score = max(risk_score, 0.30)
                risk_factors.append("long_trajectory")
            if tool_name in {"write_memory", "read_memory"} or any(
                st.tool_name in {"write_memory", "read_memory"} for st in history
            ):
                risk_score = max(risk_score, 0.25)
                risk_factors.append("memory_channel_activity")

        if use_provenance_features:
            if (
                provenance_depth >= 3
                and has_sensitive_ancestor
                and destination != "MEMORY"
            ):
                risk_score = max(risk_score, 0.60)
                risk_factors.append("deep_sensitive_provenance")

            # B4.2: Early-warning signal when a CONFIDENTIAL ancestor undergoes
            # staging/retrieval transformations (summarize, paraphrase, calculate,
            # or read_memory) BEFORE any external send occurs.
            if has_sensitive_ancestor:
                base_anc_risk = 0.30
                if sensitive_ancestor_count > 1:
                    base_anc_risk += min(0.10, 0.03 * (sensitive_ancestor_count - 1))
                risk_score = max(risk_score, base_anc_risk)
                risk_factors.append("has_sensitive_ancestor")

                if tool_name == "write_memory":
                    risk_score = max(risk_score, 0.45)
                    risk_factors.append("sensitive_memory_staging")
                elif use_transformation_depth_signal and (
                    transformation_depth >= 1 or tool_name == "read_memory"
                ):
                    eff_depth = max(1, transformation_depth)
                    depth_risk = min(0.80, 0.62 + 0.06 * (eff_depth - 1))
                    risk_score = max(risk_score, depth_risk)
                    risk_factors.append(f"transformed_sensitive_lineage_d{eff_depth}")

        if destination == "EXTERNAL" and history:
            ext_risk = (
                0.70 if (use_provenance_features and has_sensitive_ancestor) else 0.25
            )
            risk_score = max(risk_score, ext_risk)
            risk_factors.append("external_destination")

        clamped = round(min(1.0, max(0.0, risk_score)), 4)
        if not risk_factors:
            reason = "No heuristic risk factors detected."
        else:
            reason = "Risk factors detected: " + ", ".join(risk_factors)

        return RiskAssessment(
            risk_score=clamped,
            risk_factors=risk_factors,
            reason=reason,
        )
