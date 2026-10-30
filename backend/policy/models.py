from typing import Literal

from pydantic import BaseModel, Field

from backend.dataobjects.sensitivity import SensitivityLabel

PolicyDestination = Literal["INTERNAL", "EXTERNAL", "MEMORY"]
POLICY_DESTINATIONS = frozenset({"INTERNAL", "EXTERNAL", "MEMORY"})


class PolicyRule(BaseModel):
    """
    Represents one explicit SentinelAI governance rule.
    """

    rule_id: str = Field(min_length=1)
    name: str = Field(min_length=1)

    source_sensitivity: SensitivityLabel
    destination: PolicyDestination

    verdict: Literal["ALLOW", "WARN", "BLOCK"]
    reason: str = Field(min_length=1)
    priority: int | None = None


class PolicyEvaluation(BaseModel):
    """
    Represents the result of evaluating SentinelAI policy rules.
    """

    verdict: Literal["ALLOW", "WARN", "BLOCK"]
    matched_rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
