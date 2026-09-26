from backend.policy.engine import PolicyEngine


def test_confidential_to_external_is_blocked():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        source_sensitivity="CONFIDENTIAL",
        destination="EXTERNAL",
    )

    assert result.verdict == "BLOCK"
    assert len(result.matched_rules) >= 1


def test_internal_to_external_is_warning():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        source_sensitivity="INTERNAL",
        destination="EXTERNAL",
    )

    assert result.verdict == "WARN"
    assert len(result.matched_rules) >= 1


def test_public_to_external_is_allowed():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        source_sensitivity="PUBLIC",
        destination="EXTERNAL",
    )

    assert result.verdict == "ALLOW"
    assert len(result.matched_rules) >= 1


def test_confidential_to_memory_is_warning():
    engine = PolicyEngine("policies/default.yaml")

    result = engine.evaluate_action(
        source_sensitivity="CONFIDENTIAL",
        destination="MEMORY",
    )

    assert result.verdict == "WARN"
    assert len(result.matched_rules) >= 1
