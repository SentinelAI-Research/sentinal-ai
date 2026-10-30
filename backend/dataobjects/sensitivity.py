from typing import Literal, cast

SensitivityLabel = Literal["PUBLIC", "INTERNAL", "CONFIDENTIAL"]

# These labels are supported by the runtime policies. Their order is consistent
# with default/strict policy outcomes for external and memory destinations.
# RESTRICTED currently appears only in ML feature encodings and is not a
# supported runtime classification until policy semantics are defined for it.
SENSITIVITY_ORDER: dict[str, int] = {
    "PUBLIC": 0,
    "INTERNAL": 1,
    "CONFIDENTIAL": 2,
}


def most_sensitive_sensitivity(sensitivities: list[str]) -> SensitivityLabel:
    """Return the highest runtime label; reject empty or unsupported labels."""
    if not sensitivities:
        raise ValueError("At least one sensitivity label is required.")

    unknown = sorted(set(sensitivities) - set(SENSITIVITY_ORDER))
    if unknown:
        raise ValueError(f"Unsupported sensitivity label(s): {', '.join(unknown)}")

    return cast(
        SensitivityLabel,
        max(sensitivities, key=SENSITIVITY_ORDER.__getitem__),
    )
