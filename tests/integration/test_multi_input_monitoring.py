import pytest
import yaml

from backend.dataobjects.factory import create_data_object
from backend.dataobjects.memory_service import MemoryDataObjectService
from backend.dataobjects.models import DataObject
from backend.monitor.executor import ToolExecutor
from backend.monitor.monitor import SentinelMonitor
from backend.monitor.output_handler import OutputDataObjectHandler
from backend.monitor.runtime import RuntimeExecutor
from backend.monitor.service import MonitorService
from backend.policy.engine import PolicyEngine
from backend.provenance.graph import ProvenanceGraph
from backend.risk.heuristic import HeuristicRiskScorer
from backend.schemas.contracts import ToolCallProposal
from backend.trajectory.store import TrajectoryStore
from backend.monitor.models import ExecutionResult
from benchmark.runners.security_gate import SecurityGate
from backend.trajectory.models import TrajectoryStep


def build_service(sources, policy_path="policies/default.yaml"):
    graph = ProvenanceGraph()
    for object_id, sensitivity in sources.items():
        graph.add_object(
            create_data_object(
                object_id=object_id,
                source=f"test:{object_id}",
                content=object_id,
                sensitivity=sensitivity,
                transformation=None,
                parents=[],
                created_step="source",
            )
        )

    trajectory_store = TrajectoryStore()
    tool_calls = []
    tool_executor = ToolExecutor()

    def test_tool(**arguments):
        tool_calls.append(arguments)
        return "combined output"

    tool_executor.register_tool("test_tool", test_tool)
    monitor = SentinelMonitor(
        policy_engine=PolicyEngine(policy_path),
        risk_scorer=HeuristicRiskScorer(),
        provenance_graph=graph,
        trajectory_store=trajectory_store,
    )
    service = MonitorService(
        monitor=monitor,
        runtime_executor=RuntimeExecutor(trajectory_store, tool_executor),
        output_handler=OutputDataObjectHandler(graph),
    )
    return service, graph, tool_calls


def make_proposal(step_id, input_ids):
    return ToolCallProposal(
        step_id=step_id,
        agent_id="multi-input-test-agent",
        tool_name="test_tool",
        arguments={"value": "combined"},
        input_data_objects=input_ids,
    )


def test_single_input_retains_existing_allow_and_output_sensitivity():
    service, graph, calls = build_service({"source": "PUBLIC"})

    decision, result = service.process(
        make_proposal("single", ["source"]),
        destination="INTERNAL",
    )

    assert decision.verdict == "ALLOW"
    assert result.executed is True
    assert calls == [{"value": "combined"}]
    assert graph.get_object(result.output_data_object_id).sensitivity == "PUBLIC"


def test_multiple_allow_inputs_remain_allowed_and_keep_all_parents():
    service, graph, calls = build_service(
        {"public": "PUBLIC", "internal": "INTERNAL"}
    )

    decision, result = service.process(
        make_proposal("multi-allow", ["public", "internal"]),
        destination="INTERNAL",
    )

    output = graph.get_object(result.output_data_object_id)
    assert decision.verdict == "ALLOW"
    assert result.executed is True
    assert output.sensitivity == "INTERNAL"
    assert output.parents == ["public", "internal"]
    assert set(graph.get_parents(output.object_id)) == {"public", "internal"}
    assert len(calls) == 1


def test_benchmark_security_gate_checks_every_input():
    graph = ProvenanceGraph()
    for object_id, sensitivity in {
        "public": "PUBLIC",
        "confidential": "CONFIDENTIAL",
    }.items():
        graph.add_object(
            create_data_object(
                object_id=object_id,
                source=f"test:{object_id}",
                content=object_id,
                sensitivity=sensitivity,
                transformation=None,
                parents=[],
                created_step="source",
            )
        )

    step = TrajectoryStep(
        step_id="benchmark-multi-input",
        agent_id="benchmark-agent",
        session_id="benchmark-session",
        tool_name="send_message",
        arguments={},
        input_data_objects=["confidential", "public"],
    )
    gate = SecurityGate(PolicyEngine("policies/default.yaml"), HeuristicRiskScorer())

    policy_result, _ = gate.evaluate(
        step=step,
        provenance_graph=graph,
        history=[],
        destination="EXTERNAL",
    )

    assert policy_result.verdict == "BLOCK"


def test_confidential_input_followed_by_public_cannot_bypass_block():
    service, _, calls = build_service(
        {"confidential": "CONFIDENTIAL", "public": "PUBLIC"}
    )

    decision, result = service.process(
        make_proposal("confidential-then-public", ["confidential", "public"]),
        destination="EXTERNAL",
    )

    assert decision.verdict == "BLOCK"
    assert result.executed is False
    assert calls == []


def test_warn_input_followed_by_allow_input_remains_warning():
    service, _, calls = build_service(
        {"internal": "INTERNAL", "public": "PUBLIC"}
    )

    decision, result = service.process(
        make_proposal("internal-then-public", ["internal", "public"]),
        destination="EXTERNAL",
    )

    assert decision.verdict == "WARN"
    assert result.executed is False
    assert calls == []


@pytest.mark.parametrize(
    "input_ids",
    [["confidential", "public"], ["public", "confidential"]],
)
def test_input_order_does_not_weaken_decision_or_output_sensitivity(input_ids):
    service, _, calls = build_service(
        {"confidential": "CONFIDENTIAL", "public": "PUBLIC"}
    )

    decision, result = service.process(
        make_proposal(f"order-{input_ids[0]}", input_ids),
        destination="EXTERNAL",
    )

    assert decision.verdict == "BLOCK"
    assert result.executed is False
    assert calls == []

    allow_service, allow_graph, _ = build_service(
        {"confidential": "CONFIDENTIAL", "public": "PUBLIC"}
    )
    allow_decision, allow_result = allow_service.process(
        make_proposal(f"allow-order-{input_ids[0]}", input_ids),
        destination="INTERNAL",
    )
    assert allow_decision.verdict == "ALLOW"
    assert (
        allow_graph.get_object(allow_result.output_data_object_id).sensitivity
        == "CONFIDENTIAL"
    )


@pytest.mark.parametrize(
    ("sensitivity", "destination", "expected_verdict"),
    [
        ("INTERNAL", "EXTERNAL", "WARN"),
        ("CONFIDENTIAL", "EXTERNAL", "BLOCK"),
    ],
)
def test_warn_and_block_never_invoke_tool(sensitivity, destination, expected_verdict):
    service, _, calls = build_service({"source": sensitivity})

    decision, result = service.process(
        make_proposal(f"no-run-{sensitivity}", ["source"]),
        destination=destination,
    )

    assert decision.verdict == expected_verdict
    assert result.executed is False
    assert calls == []


def test_missing_input_is_rejected_before_tool_invocation():
    service, graph, calls = build_service({"known": "PUBLIC"})

    with pytest.raises(ValueError, match="missing-input"):
        service.process(
            make_proposal("missing-input", ["known", "missing-input"]),
            destination="INTERNAL",
        )

    assert calls == []
    assert service.monitor.trajectory_store.count() == 0
    assert not graph.has_object("missing-input-output")


def test_missing_parent_does_not_create_partial_output_object():
    service, graph, calls = build_service({"known": "PUBLIC"})

    with pytest.raises(ValueError, match="missing-parent"):
        service.output_handler.create_output_object(
            proposal=make_proposal("partial-output", ["known", "missing-parent"]),
            execution_result=ExecutionResult(
                executed=True,
                tool_name="test_tool",
                output="should not be registered",
            ),
        )

    assert calls == []
    assert not graph.has_object("partial-output-output")


def test_unknown_sensitivity_fails_closed_before_tool_invocation():
    service, graph, calls = build_service({"known": "PUBLIC"})
    source = graph.get_object("known")
    object.__setattr__(source, "sensitivity", "TOP_SECRET")

    with pytest.raises(ValueError, match="Unsupported sensitivity label"):
        service.process(
            make_proposal("unknown-sensitivity", ["known"]),
            destination="EXTERNAL",
        )

    assert calls == []


def test_no_input_is_rejected_deterministically_before_tool_invocation():
    service, _, calls = build_service({})

    with pytest.raises(ValueError, match="has no input data objects"):
        service.process(make_proposal("no-input", []), destination="INTERNAL")

    assert calls == []


def test_duplicate_parent_ids_are_rejected_before_tool_invocation():
    service, graph, calls = build_service({"source": "PUBLIC"})

    with pytest.raises(ValueError, match="Duplicate input DataObject IDs"):
        service.process(
            make_proposal("duplicate-parent", ["source", "source"]),
            destination="INTERNAL",
        )

    assert calls == []
    assert service.monitor.trajectory_store.count() == 0
    assert not graph.has_object("duplicate-parent-output")


def test_duplicate_step_id_does_not_execute_twice_or_create_another_output():
    service, graph, calls = build_service({"source": "PUBLIC"})
    proposal = make_proposal("duplicate-step", ["source"])

    first_decision, first_result = service.process(proposal, destination="INTERNAL")
    assert first_decision.verdict == "ALLOW"
    assert first_result.executed is True
    assert len(calls) == 1

    with pytest.raises(ValueError, match="already exists"):
        service.process(proposal, destination="INTERNAL")

    assert len(calls) == 1
    assert service.monitor.trajectory_store.count() == 1
    assert graph.has_object("duplicate-step-output")


def test_memory_objects_validate_all_parents_and_keep_highest_sensitivity():
    service, graph, _ = build_service(
        {"public": "PUBLIC", "confidential": "CONFIDENTIAL"}
    )
    memory_service = MemoryDataObjectService(graph)

    memory_object = memory_service.create_memory_object(
        object_id="memory-output",
        content="combined",
        parent_object_ids=["public", "confidential"],
        sensitivity="PUBLIC",
        created_step="memory-write",
    )
    assert memory_object.sensitivity == "CONFIDENTIAL"
    assert set(graph.get_parents(memory_object.object_id)) == {
        "public",
        "confidential",
    }

    with pytest.raises(ValueError, match="Duplicate input DataObject IDs"):
        memory_service.create_memory_object(
            object_id="duplicate-memory",
            content="combined",
            parent_object_ids=["public", "public"],
            sensitivity="PUBLIC",
            created_step="memory-write-duplicate",
        )
    assert not graph.has_object("duplicate-memory")

    with pytest.raises(ValueError, match="unknown-memory-parent"):
        memory_service.create_memory_object(
            object_id="missing-memory-parent",
            content="combined",
            parent_object_ids=["public", "unknown-memory-parent"],
            sensitivity="PUBLIC",
            created_step="memory-write-missing",
        )
    assert not graph.has_object("missing-memory-parent")


def test_data_object_rejects_unsupported_runtime_sensitivity():
    with pytest.raises(ValueError):
        DataObject(
            object_id="restricted-runtime-object",
            source="test",
            sensitivity="RESTRICTED",
        )


def test_policy_evaluation_rejects_unsupported_sensitivity_and_destination():
    engine = PolicyEngine("policies/default.yaml")

    with pytest.raises(ValueError, match="Unsupported sensitivity label"):
        engine.evaluate_action("RESTRICTED", "EXTERNAL")

    with pytest.raises(ValueError, match="Unsupported policy destination"):
        engine.evaluate_action("PUBLIC", "REMOTE")


def test_policy_configuration_rejects_unknown_verdict_at_load_time(tmp_path):
    config_path = tmp_path / "invalid-policy.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "rules": [
                    {
                        "rule_id": "invalid-verdict",
                        "name": "Invalid verdict",
                        "source_sensitivity": "PUBLIC",
                        "destination": "EXTERNAL",
                        "verdict": "DENY",
                        "reason": "DENY is not a supported verdict.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid policy rule at index 0"):
        PolicyEngine(str(config_path))


@pytest.mark.parametrize(
    "invalid_rule",
    [
        {"rule_id": "bad-sensitivity", "name": "Bad sensitivity", "source_sensitivity": "RESTRICTED", "destination": "EXTERNAL", "verdict": "BLOCK", "reason": "unsupported runtime class"},
        {"rule_id": "bad-destination", "name": "Bad destination", "source_sensitivity": "PUBLIC", "destination": "REMOTE", "verdict": "ALLOW", "reason": "unknown destination"},
        {"rule_id": "", "name": "Missing ID", "source_sensitivity": "PUBLIC", "destination": "EXTERNAL", "verdict": "ALLOW", "reason": "empty ID"},
        {"rule_id": "unknown-field", "name": "Unknown field", "source_sensitivity": "PUBLIC", "destination": "EXTERNAL", "verdict": "ALLOW", "reason": "unknown field", "unexpected": True},
    ],
)
def test_malformed_policy_rules_are_rejected_at_load_time(tmp_path, invalid_rule):
    config_path = tmp_path / "invalid-policy.yaml"
    config_path.write_text(
        yaml.safe_dump({"rules": [invalid_rule]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="policy rule at index 0"):
        PolicyEngine(str(config_path))


def test_policy_configuration_rejects_duplicate_rule_ids(tmp_path):
    rule = {
        "rule_id": "duplicate",
        "name": "Duplicate ID",
        "source_sensitivity": "PUBLIC",
        "destination": "EXTERNAL",
        "verdict": "ALLOW",
        "reason": "valid rule",
    }
    config_path = tmp_path / "duplicate-policy.yaml"
    config_path.write_text(
        yaml.safe_dump({"rules": [rule, {**rule, "name": "Duplicate again"}]}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate policy rule_id"):
        PolicyEngine(str(config_path))


@pytest.mark.parametrize(
    ("policy_path", "sensitivity", "destination", "expected_verdict"),
    [
        ("policies/default.yaml", "PUBLIC", "EXTERNAL", "ALLOW"),
        ("policies/default.yaml", "INTERNAL", "EXTERNAL", "WARN"),
        ("policies/default.yaml", "CONFIDENTIAL", "EXTERNAL", "BLOCK"),
        ("policies/strict.yaml", "INTERNAL", "EXTERNAL", "BLOCK"),
        ("policies/strict.yaml", "INTERNAL", "MEMORY", "WARN"),
        ("policies/strict.yaml", "CONFIDENTIAL", "MEMORY", "BLOCK"),
        ("policies/strict.yaml", "PUBLIC", "EXTERNAL", "ALLOW"),
    ],
)
def test_existing_default_and_strict_policy_verdicts_are_preserved(
    policy_path, sensitivity, destination, expected_verdict
):
    service, _, calls = build_service({"source": sensitivity}, policy_path)

    decision, result = service.process(
        make_proposal(f"policy-{sensitivity}-{destination}", ["source"]),
        destination=destination,
    )

    assert decision.verdict == expected_verdict
    assert result.executed is (expected_verdict == "ALLOW")
    assert len(calls) == int(expected_verdict == "ALLOW")
