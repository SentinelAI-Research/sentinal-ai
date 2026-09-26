from pathlib import Path

from backend.policy.engine import PolicyEngine
from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.store import TrajectoryStore
from benchmark.runners.execution_service import ScenarioExecutionService
from benchmark.runners.provenance_recorder import ProvenanceRecorder
from benchmark.runners.security_gate import SecurityGate
from benchmark.scenarios.memory_mediated import (
    create_memory_mediated_scenario,
)
from backend.risk.heuristic import HeuristicRiskScorer


def test_memory_mediated_external_action_is_blocked():
    trajectory_store = TrajectoryStore()

    provenance_graph = ProvenanceGraph()

    provenance_recorder = ProvenanceRecorder(provenance_graph)

    service = ScenarioExecutionService(
        trajectory_store=trajectory_store,
        provenance_recorder=provenance_recorder,
    )

    scenario = create_memory_mediated_scenario()

    file_path = str(Path("data/sample/confidential_document.txt"))

    execution = service.execute_memory_path(
        scenario=scenario,
        file_path=file_path,
        sensitivity="CONFIDENTIAL",
    )

    final_step = execution.trajectory[-1]

    policy_engine = PolicyEngine("policies/default.yaml")

    risk_scorer = HeuristicRiskScorer()

    gate = SecurityGate(
        policy_engine=policy_engine,
        risk_scorer=risk_scorer,
    )

    history = execution.trajectory[:-1]

    policy_result, risk_result = gate.evaluate(
        step=final_step,
        provenance_graph=provenance_graph,
        history=history,
        destination="EXTERNAL",
    )
    
    assert policy_result.verdict == "BLOCK"

    assert (
        "CONFIDENTIAL_TO_EXTERNAL"
        in policy_result.matched_rules
    )

    assert risk_result.risk_score == 1.0

    assert "policy_block" in risk_result.risk_factors
