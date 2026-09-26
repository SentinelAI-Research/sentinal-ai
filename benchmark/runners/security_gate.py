from backend.policy.engine import PolicyEngine
from backend.risk.heuristic import HeuristicRiskScorer
from backend.trajectory.models import TrajectoryStep

"""
Creating boundary between trajectory and policy engine : 

Trajectory
      ↓
SecurityGate
      ↓
PolicyEngine
      ↓
ALLOW / WARN / BLOCK
"""


class SecurityGate:
    def __init__(self, policy_engine: PolicyEngine, risk_scorer: HeuristicRiskScorer):
        self.policy_engine = policy_engine
        self.risk_scorer = risk_scorer

    def evaluate(
        self,
        step: TrajectoryStep,
        provenance_graph,
        history: list[TrajectoryStep],
        destination: str,
    ):
        if not step.input_data_objects:
            raise ValueError(f"Step {step.step_id} has no input data objects.")

        input_object_id = step.input_data_objects[-1]

        data_object = provenance_graph.get_object(input_object_id)

        policy_result = self.policy_engine.evaluate_action(
            source_sensitivity=data_object.sensitivity,
            destination=destination,
        )

        provenance_depth = provenance_graph.get_depth(input_object_id)

        risk_result = self.risk_scorer.assess(
            policy_result=policy_result,
            history=history,
            provenance_depth=provenance_depth,
            destination=destination,
        )

        return policy_result, risk_result
