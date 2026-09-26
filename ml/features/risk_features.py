from dataclasses import dataclass


@dataclass
class RiskFeatures:
    """
    The ML model cannot directly understand:

    This action has a deep provenance chain and eventually sends derived confidential information externally.

    We need to convert that information into numeric features.
    The model can then learn relationships between these features and the security outcome.
    """
    trajectory_length: int
    provenance_depth: int
    external_destination: int
    policy_warning: int
    policy_block: int
    transformation_count: int


def extract_features(
    trajectory_length: int,
    provenance_depth: int,
    destination: str,
    policy_verdict: str,
    transformation_count: int,
) -> RiskFeatures:
    return RiskFeatures(
        trajectory_length=trajectory_length,
        provenance_depth=provenance_depth,
        external_destination=int(destination == "EXTERNAL"),
        policy_warning=int(policy_verdict == "WARN"),
        policy_block=int(policy_verdict == "BLOCK"),
        transformation_count=transformation_count,
    )