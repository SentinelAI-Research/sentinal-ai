from pydantic import BaseModel, Field


class RiskAssessment(BaseModel):
    """
    Represents the result of SentinelAI risk assessment.
    0.0 → very low observed risk
    1.0 → very high observed risk
    """

    risk_score: float = Field(ge=0.0, le=1.0)

    risk_factors: list[str] = Field(
        default_factory=list
    )

    reason: str