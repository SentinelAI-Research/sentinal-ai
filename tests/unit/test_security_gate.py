from pathlib import Path

from backend.dataobjects.models import DataObject
from backend.policy.engine import PolicyEngine
from backend.provenance.graph import ProvenanceGraph
from backend.trajectory.models import TrajectoryStep
from benchmark.runners.security_gate import SecurityGate
from backend.risk.heuristic import HeuristicRiskScorer


def test_confidential_external_action_is_blocked():
    policy_path = Path("policies/default.yaml")

    policy_engine = PolicyEngine(str(policy_path))

    risk_scorer = HeuristicRiskScorer()

    gate = SecurityGate(
        policy_engine=policy_engine,
        risk_scorer=risk_scorer,
    )

    provenance_graph = ProvenanceGraph()

    data_object = DataObject(
        object_id="memory-retrieval",
        source="memory",
        sensitivity="CONFIDENTIAL",
        parents=[],
        transformation="read_memory",
        content_hash="hash-memory",
        embedding_ref=None,
        created_step="step-4",
    )

    provenance_graph.add_object(data_object)

    step = TrajectoryStep(
        step_id="step-5",
        agent_id="benchmark-agent",
        session_id="session-1",
        tool_name="send_message",
        arguments={
            "recipient": "external@example.com",
        },
        input_data_objects=["memory-retrieval"],
    )

    policy_result, risk_result = gate.evaluate(
        step=step,
        provenance_graph=provenance_graph,
        history=[],
        destination="EXTERNAL",
    )

    assert policy_result.verdict == "BLOCK"

    assert "CONFIDENTIAL_TO_EXTERNAL" in policy_result.matched_rules

    assert risk_result.risk_score == 1.0
    assert "policy_block" in risk_result.risk_factors
