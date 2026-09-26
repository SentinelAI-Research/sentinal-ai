from dataclasses import dataclass

"""
Scenario describes what security situation we are testing
For eg.
scenario_id: summary-leak-001
scenario_family: summary_leakage
description: Confidential document is summarized and later prepared for external transmission
expected_label: VIOLATION
"""

@dataclass
class SecurityScenario:
    scenario_id: str
    scenario_family: str
    description: str
    expected_label: str