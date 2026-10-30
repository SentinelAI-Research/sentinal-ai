from backend.policy.engine import PolicyEngine
from backend.risk.heuristic import HeuristicRiskScorer
from backend.trajectory.models import TrajectoryStep
from backend.dataobjects.sensitivity import most_sensitive_sensitivity

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

        input_objects = provenance_graph.resolve_objects(step.input_data_objects)
        most_sensitive_sensitivity(
            [data_object.sensitivity for data_object in input_objects]
        )
        policy_results = [
            self.policy_engine.evaluate_action(
                source_sensitivity=data_object.sensitivity,
                destination=destination,
            )
            for data_object in input_objects
        ]
        policy_result = self.policy_engine.combine_evaluations(policy_results)

        provenance_depth = max(
            provenance_graph.get_depth(object_id)
            for object_id in step.input_data_objects
        )

        risk_result = self.risk_scorer.assess(
            policy_result=policy_result,
            history=history,
            provenance_depth=provenance_depth,
            destination=destination,
        )

        return policy_result, risk_result
