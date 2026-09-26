from dataclasses import dataclass

from ml.data.labels import VALID_LABELS


@dataclass
class TrainingRecord:
    """
    One record of eventual training dataset
    """
    trajectory_length: int
    provenance_depth: int
    external_destination: int
    policy_warning: int
    policy_block: int
    transformation_count: int
    label: str
    scenario_id: str
    scenario_family: str
    source_id: str

    def validate(self) -> None:
        """
        Add validation for training records
        This prevents bad data from silently entering the ML pipeline.
        """
        if self.trajectory_length < 0:
            raise ValueError("trajectory_length cannot be negative.")

        if self.provenance_depth < 0:
            raise ValueError("provenance_depth cannot be negative.")

        if self.transformation_count < 0:
            raise ValueError("transformation_count cannot be negative.")

        if self.external_destination not in {0, 1}:
            raise ValueError("external_destination must be 0 or 1.")

        if self.policy_warning not in {0, 1}:
            raise ValueError("policy_warning must be 0 or 1.")

        if self.policy_block not in {0, 1}:
            raise ValueError("policy_block must be 0 or 1.")

        if self.label not in VALID_LABELS:
            raise ValueError(f"Invalid label: {self.label}")

        if not self.scenario_id:
            raise ValueError("scenario_id cannot be empty.")

        if not self.scenario_family:
            raise ValueError("scenario_family cannot be empty.")

        if not self.source_id:
            raise ValueError("source_id cannot be empty.")