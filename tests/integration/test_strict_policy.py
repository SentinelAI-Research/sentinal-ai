from backend.policy.engine import PolicyEngine


def test_strict_policy_blocks_internal_external():
    engine = PolicyEngine("policies/strict.yaml")

    result = engine.evaluate_action(
        source_sensitivity="INTERNAL",
        destination="EXTERNAL",
    )

    assert result.verdict == "BLOCK"


def test_strict_policy_blocks_confidential_memory():
    engine = PolicyEngine("policies/strict.yaml")

    result = engine.evaluate_action(
        source_sensitivity="CONFIDENTIAL",
        destination="MEMORY",
    )

    assert result.verdict == "BLOCK"


def test_strict_policy_allows_public_external():
    engine = PolicyEngine("policies/strict.yaml")

    result = engine.evaluate_action(
        source_sensitivity="PUBLIC",
        destination="EXTERNAL",
    )

    assert result.verdict == "ALLOW"
