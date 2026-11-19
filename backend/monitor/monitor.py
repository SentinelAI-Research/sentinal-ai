from __future__ import annotations

from typing import Any

import joblib

from backend.dataobjects.sensitivity import most_sensitive_sensitivity
from backend.monitor.config import MonitorConfig
from backend.policy.engine import PolicyEngine
from backend.policy.models import PolicyEvaluation
from backend.provenance.graph import ProvenanceGraph
from backend.risk.heuristic import HeuristicRiskScorer
from backend.schemas.contracts import Decision, ToolCallProposal
from backend.trajectory.models import TrajectoryStep
from backend.trajectory.store import TrajectoryStore


class SentinelMonitor:
    """
    Central SentinelAI security monitor supporting toggleable configurations
    (C0, C1, C2, C2b, C3, C4) and Risk-Verdict Fusion (B4.1 & B5).
    """

    def __init__(
        self,
        policy_engine: PolicyEngine,
        risk_scorer: HeuristicRiskScorer,
        provenance_graph: ProvenanceGraph,
        trajectory_store: TrajectoryStore,
        config: MonitorConfig | None = None,
    ):
        self.policy_engine = policy_engine
        self.risk_scorer = risk_scorer
        self.provenance_graph = provenance_graph
        self.trajectory_store = trajectory_store
        self.config = config or MonitorConfig.from_preset("C4")
        self._session_confidential_read = False
        self.last_provenance_features: dict[str, Any] = {}
        self._ml_bundle: dict[str, Any] | None = None

        if self.config.risk_model_path:
            loaded = joblib.load(self.config.risk_model_path)
            if isinstance(loaded, dict) and "model" in loaded:
                self._ml_bundle = loaded
            else:
                self._ml_bundle = {"model": loaded, "feature_names": None}

    def reset_session(self) -> None:
        self._session_confidential_read = False
        self.last_provenance_features = {}

    def _compute_transformation_depth(self, object_id: str) -> int:
        if not self.provenance_graph.has_object(object_id):
            return 0
        obj = self.provenance_graph.get_object(object_id)
        parents = list(self.provenance_graph.get_parents(object_id))
        if not parents:
            return 0
        trans = str(getattr(obj, "transformation", "") or "")
        # Each write_memory -> read_memory hop counts as 1 transformation depth
        step_cost = (
            1 if trans in {"summarize", "paraphrase", "calculate", "read_memory"} else 0
        )
        return step_cost + max(
            self._compute_transformation_depth(pid) for pid in parents
        )

    def extract_provenance_features(
        self,
        input_object_ids: list[str],
    ) -> tuple[dict[str, Any], list[str]]:
        provenance_trace = list(dict.fromkeys(input_object_ids))
        sensitive_ancestors: set[str] = set()
        max_trans_depth = 0

        for object_id in input_object_ids:
            if not self.provenance_graph.has_object(object_id):
                continue
            obj = self.provenance_graph.get_object(object_id)
            if str(obj.sensitivity).upper() == "CONFIDENTIAL":
                sensitive_ancestors.add(object_id)

            for ancestor_id in sorted(self.provenance_graph.get_ancestors(object_id)):
                if ancestor_id not in provenance_trace:
                    provenance_trace.append(ancestor_id)
                anc = self.provenance_graph.get_object(ancestor_id)
                if str(anc.sensitivity).upper() == "CONFIDENTIAL":
                    sensitive_ancestors.add(ancestor_id)

            max_trans_depth = max(
                max_trans_depth, self._compute_transformation_depth(object_id)
            )

        prov_depth = max(
            (
                self.provenance_graph.get_depth(oid)
                for oid in input_object_ids
                if self.provenance_graph.has_object(oid)
            ),
            default=0,
        )

        features = {
            "has_sensitive_ancestor": int(len(sensitive_ancestors) > 0),
            "sensitive_ancestor_count": len(sensitive_ancestors),
            "provenance_depth": prov_depth,
            "transformation_depth": max_trans_depth,
            "num_parents": len(input_object_ids),
        }
        return features, provenance_trace

    def _predict_ml_probability(
        self,
        *,
        tool_name: str,
        destination: str,
        input_sensitivities: list[str],
        prov_features: dict[str, Any],
        trajectory_length: int,
        memory_used: bool,
    ) -> float | None:
        if self._ml_bundle is None:
            return None
        model = self._ml_bundle["model"]
        feature_names: list[str] | None = self._ml_bundle.get("feature_names")

        upper_sens = {str(s).upper() for s in input_sensitivities}
        feature_map: dict[str, float] = {
            "is_external": float(str(destination).upper().startswith("EXTERNAL")),
            "has_confidential_input": float("CONFIDENTIAL" in upper_sens),
            "has_internal_input": float("INTERNAL" in upper_sens),
            "is_send_message": float(tool_name == "send_message"),
            "is_memory_tool": float(tool_name in {"write_memory", "read_memory"}),
            "trajectory_length": float(trajectory_length),
            "memory_used": float(memory_used),
            "num_parents": float(prov_features.get("num_parents", 1)),
            "has_sensitive_ancestor": float(
                prov_features.get("has_sensitive_ancestor", 0)
                if self.config.use_provenance_features
                else 0
            ),
            "sensitive_ancestor_count": float(
                prov_features.get("sensitive_ancestor_count", 0)
                if self.config.use_provenance_features
                else 0
            ),
            "provenance_depth": float(
                prov_features.get("provenance_depth", 0)
                if self.config.use_provenance_features
                else 0
            ),
            "transformation_depth": float(
                prov_features.get("transformation_depth", 0)
                if self.config.use_provenance_features
                else 0
            ),
        }

        if feature_names:
            row = [[feature_map.get(name, 0.0) for name in feature_names]]
        else:
            row = [list(feature_map.values())]

        if hasattr(model, "predict_proba"):
            return round(float(model.predict_proba(row)[0][1]), 4)
        return round(float(model.predict(row)[0]), 4)

    def evaluate(
        self,
        proposal: ToolCallProposal,
        destination: str,
    ) -> Decision:
        if not proposal.input_data_objects:
            raise ValueError(f"Proposal {proposal.step_id} has no input data objects.")

        output_object_id = f"{proposal.step_id}-output"
        if self.provenance_graph.has_object(output_object_id):
            raise ValueError(f"Data object already exists: {output_object_id}")

        input_objects = self.provenance_graph.resolve_objects(
            proposal.input_data_objects
        )
        sensitivities = [data_object.sensitivity for data_object in input_objects]
        most_sensitive_sensitivity(sensitivities)

        # Track session-level confidential read for C2b session taint baseline
        if any(str(s).upper() == "CONFIDENTIAL" for s in sensitivities):
            self._session_confidential_read = True

        prov_features, provenance_trace = self.extract_provenance_features(
            list(proposal.input_data_objects)
        )
        self.last_provenance_features = dict(prov_features)
        provenance_depth = int(prov_features["provenance_depth"])

        session_id = f"session-{proposal.agent_id}"
        step = TrajectoryStep(
            step_id=proposal.step_id,
            agent_id=proposal.agent_id,
            session_id=session_id,
            tool_name=proposal.tool_name,
            arguments=proposal.arguments,
            input_data_objects=proposal.input_data_objects,
        )
        self.trajectory_store.add_step(step)
        history = self.trajectory_store.get_history(
            session_id,
            proposal.step_id,
        )

        # C0: Unmonitored baseline (always ALLOW; still logs trajectory & provenance)
        if not self.config.enabled:
            return Decision(
                step_id=proposal.step_id,
                verdict="ALLOW",
                risk_score=0.0,
                risk_probability=None,
                policy_hits=[],
                provenance_trace=provenance_trace,
                reason="C0 unmonitored baseline (always ALLOW).",
            )

        # 1. Policy Evaluation
        if self.config.use_policy_rules:
            policy_results = [
                self.policy_engine.evaluate_action(
                    source_sensitivity=sensitivity,
                    destination=destination,
                )
                for sensitivity in sensitivities
            ]
            policy_result = self.policy_engine.combine_evaluations(policy_results)
        else:
            policy_result = PolicyEvaluation(
                verdict="ALLOW",
                matched_rules=[],
                reason="Policy rules disabled in MonitorConfig.",
            )

        # 2. Risk Evaluation
        memory_used = proposal.tool_name in {"write_memory", "read_memory"} or any(
            st.tool_name in {"write_memory", "read_memory"} for st in history
        )
        risk_result = self.risk_scorer.assess(
            policy_result=policy_result,
            history=history,
            provenance_depth=provenance_depth,
            destination=destination,
            tool_name=proposal.tool_name,
            has_sensitive_ancestor=bool(prov_features["has_sensitive_ancestor"]),
            sensitive_ancestor_count=int(prov_features["sensitive_ancestor_count"]),
            transformation_depth=int(prov_features["transformation_depth"]),
            use_provenance_features=self.config.use_provenance_features,
            use_transformation_depth_signal=self.config.use_transformation_depth_signal,
            use_trajectory_features=self.config.use_trajectory_features,
        )

        ml_prob = self._predict_ml_probability(
            tool_name=proposal.tool_name,
            destination=destination,
            input_sensitivities=sensitivities,
            prov_features=prov_features,
            trajectory_length=len(history),
            memory_used=memory_used,
        )
        effective_risk = ml_prob if ml_prob is not None else risk_result.risk_score

        # 3. Verdict Fusion Rule (B4.1 & B5):
        #    - Policy BLOCK always wins
        #    - C2b Session Taint: BLOCK any external send once CONFIDENTIAL was read in session
        #    - Otherwise if risk_fusion enabled and effective_risk >= warn_threshold -> WARN
        #    - Otherwise ALLOW
        is_external = str(destination).upper().startswith("EXTERNAL")
        matched_rules = list(policy_result.matched_rules)

        if policy_result.verdict == "BLOCK":
            final_verdict = "BLOCK"
            final_reason = risk_result.reason
        elif (
            self.config.session_taint_mode
            and self._session_confidential_read
            and is_external
        ):
            final_verdict = "BLOCK"
            matched_rules.append("SESSION_TAINT_EXTERNAL_BLOCK")
            final_reason = "C2b Session Taint: external send blocked after confidential read in session."
        elif (
            self.config.use_risk_fusion and effective_risk >= self.config.warn_threshold
        ):
            final_verdict = "WARN"
            final_reason = (
                f"Risk score {effective_risk:.3f} >= warn_threshold "
                f"{self.config.warn_threshold:.3f} ({risk_result.reason})."
            )
        elif policy_result.verdict == "WARN" and destination != "MEMORY":
            final_verdict = "WARN"
            final_reason = risk_result.reason
        else:
            final_verdict = "ALLOW"
            final_reason = risk_result.reason

        return Decision(
            step_id=proposal.step_id,
            verdict=final_verdict,
            risk_score=effective_risk,
            risk_probability=ml_prob,
            policy_hits=matched_rules,
            provenance_trace=provenance_trace,
            reason=final_reason,
        )
