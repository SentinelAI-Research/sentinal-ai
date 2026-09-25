from pathlib import Path

import yaml

from backend.policy.models import PolicyEvaluation, PolicyRule


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
            raise FileNotFoundError(
                f"Policy file not found: {self.policy_path}"
            )

        with self.policy_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        return [
            PolicyRule(**rule)
            for rule in config.get("rules", [])
        ]

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
            key=lambda rule: verdict_priority.get(rule.verdict, -1),
        ).verdict

    def evaluate_action(
        self,
        source_sensitivity: str,
        destination: str,
    ) -> PolicyEvaluation:
        """
        Evaluate an action and return a structured policy result.
        """

        rules = self.evaluate(
            source_sensitivity,
            destination,
        )

        return PolicyEvaluation(
            verdict=self.get_verdict(rules),
            matched_rules=[rule.rule_id for rule in rules],
            reasons=[rule.reason for rule in rules],
        )