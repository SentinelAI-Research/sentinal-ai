from pathlib import Path

import yaml
from pydantic import ValidationError

from backend.dataobjects.sensitivity import most_sensitive_sensitivity
from backend.policy.models import (
    POLICY_DESTINATIONS,
    PolicyEvaluation,
    PolicyRule,
)

POLICY_RULE_FIELDS = {
    "rule_id",
    "name",
    "source_sensitivity",
    "destination",
    "verdict",
    "reason",
    "priority",
}


class PolicyEngine:
    """
    Evaluates explicit SentinelAI policy rules.
    """

    def __init__(self, policy_path: str):
        self.policy_path = Path(policy_path)
        self.rules = self._load_rules()

    def _load_rules(self) -> list[PolicyRule]:
        """Load policy rules from YAML."""

        if not self.policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {self.policy_path}")

        with self.policy_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict) or not isinstance(config.get("rules"), list):
            raise ValueError("Policy configuration must contain a 'rules' list.")

        rules: list[PolicyRule] = []
        rule_ids: set[str] = set()
        for index, raw_rule in enumerate(config["rules"]):
            if not isinstance(raw_rule, dict):
                raise ValueError(f"Policy rule at index {index} must be a mapping.")
            unknown_fields = set(raw_rule) - POLICY_RULE_FIELDS
            if unknown_fields:
                names = ", ".join(sorted(unknown_fields))
                raise ValueError(
                    f"Unknown field(s) in policy rule at index {index}: {names}"
                )
            try:
                rule = PolicyRule(**raw_rule)
            except ValidationError as exc:
                raise ValueError(f"Invalid policy rule at index {index}.") from exc

            if rule.rule_id in rule_ids:
                raise ValueError(f"Duplicate policy rule_id: {rule.rule_id}")
            rule_ids.add(rule.rule_id)
            rules.append(rule)

        return rules

    def evaluate(
        self,
        source_sensitivity: str,
        destination: str,
    ) -> list[PolicyRule]:
        """
        Return all policy rules matching the source sensitivity
        and destination.
        """

        return [
            rule
            for rule in self.rules
            if rule.source_sensitivity == source_sensitivity
            and rule.destination == destination
        ]

    def get_verdict(self, rules: list[PolicyRule]) -> str:
        """
        Return the effective verdict from matching policy rules.

        BLOCK takes priority over WARN, which takes priority
        over ALLOW.
        """

        if not rules:
            return "ALLOW"

        verdict_priority = {
            "ALLOW": 0,
            "WARN": 1,
            "BLOCK": 2,
        }

        return max(
            rules,
            key=lambda rule: verdict_priority[rule.verdict],
        ).verdict

    def evaluate_action(
        self,
        source_sensitivity: str,
        destination: str,
    ) -> PolicyEvaluation:
        """
        Evaluate an action and return a structured policy result.
        """

        most_sensitive_sensitivity([source_sensitivity])
        if destination not in POLICY_DESTINATIONS:
            raise ValueError(f"Unsupported policy destination: {destination}")

        rules = self.evaluate(
            source_sensitivity,
            destination,
        )

        return PolicyEvaluation(
            verdict=self.get_verdict(rules),
            matched_rules=[rule.rule_id for rule in rules],
            reasons=[rule.reason for rule in rules],
        )

    def combine_evaluations(
        self,
        evaluations: list[PolicyEvaluation],
    ) -> PolicyEvaluation:
        """Combine source evaluations with BLOCK > WARN > ALLOW precedence."""
        if not evaluations:
            raise ValueError("At least one policy evaluation is required.")

        verdict_priority = {
            "ALLOW": 0,
            "WARN": 1,
            "BLOCK": 2,
        }
        unknown_verdicts = sorted(
            {result.verdict for result in evaluations} - set(verdict_priority)
        )
        if unknown_verdicts:
            raise ValueError(
                f"Unknown policy verdict(s): {', '.join(unknown_verdicts)}"
            )

        return PolicyEvaluation(
            verdict=max(
                (result.verdict for result in evaluations),
                key=verdict_priority.__getitem__,
            ),
            matched_rules=list(
                dict.fromkeys(
                    rule_id
                    for result in evaluations
                    for rule_id in result.matched_rules
                )
            ),
            reasons=list(
                dict.fromkeys(
                    reason for result in evaluations for reason in result.reasons
                )
            ),
        )
