from backend.policy.models import PolicyEvaluation
from backend.trajectory.models import TrajectoryStep
from ml.data.records import TrainingRecord

"""
Connect Real Trajectories to ML Training Records
TrajectoryStore
      ↓
TrajectoryStep[]
      ↓
trajectory_to_training_record()
      ↓
TrainingRecord
"""
def trajectory_to_training_record(
    history: list[TrajectoryStep],
    provenance_depth: int,
    destination: str,
    policy_result: PolicyEvaluation,
    transformation_count: int,
    label: str,
    scenario_id: str,
    scenario_family: str,
    source_id: str,
) -> TrainingRecord:
    record = TrainingRecord(
        trajectory_length=len(history),
        provenance_depth=provenance_depth,
        external_destination=int(destination == "EXTERNAL"),
        policy_warning=int(policy_result.verdict == "WARN"),
        policy_block=int(policy_result.verdict == "BLOCK"),
        transformation_count=transformation_count,
        label=label,
        scenario_id=scenario_id,
        scenario_family=scenario_family,
        source_id=source_id,
    )

    record.validate()

    return record