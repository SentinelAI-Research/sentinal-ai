from pydantic import BaseModel, Field


class PolicyRule(BaseModel):
    """
    Represents one explicit SentinelAI governance rule.
    """

    rule_id: str
    name: str

    source_sensitivity: str
    destination: str

    verdict: str
    reason: str

class PolicyEvaluation(BaseModel):
    """
    Represents the result of evaluating SentinelAI policy rules.
    """

    verdict: str
    matched_rules: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)