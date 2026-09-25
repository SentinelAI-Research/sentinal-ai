from backend.policy.engine import PolicyEngine


def test_default_policy_loads():
    engine = PolicyEngine("policies/default.yaml")

    assert len(engine.rules) == 4


def test_confidential_external_is_blocked():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        "CONFIDENTIAL",
        "EXTERNAL",
    )

    assert result.verdict == "BLOCK"
    assert "CONFIDENTIAL_TO_EXTERNAL" in result.matched_rules


def test_public_external_is_allowed():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        "PUBLIC",
        "EXTERNAL",
    )

    assert result.verdict == "ALLOW"