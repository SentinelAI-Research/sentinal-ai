import pytest

from backend.agent.demo_scenarios import (
    BLOCKED_SCENARIO_ID,
    SAFE_SCENARIO_ID,
    run_demo_scenario,
)
from backend.agent.deterministic_loop import (
    DeterministicAgentLoop,
    PlannedAction,
)
from backend.dataobjects.registry import DataObjectRegistry
from backend.monitor.factory import create_sentinel_runtime


def test_safe_demo_executes_in_order_and_records_provenance():
    result = run_demo_scenario(SAFE_SCENARIO_ID)

    assert result.status == "completed"
    assert [step.tool_name for step in result.trace] == [
        "read_file",
        "summarize",
        "paraphrase",
    ]
    assert all(step.verdict == "ALLOW" for step in result.trace)
    assert all(step.execution_status == "succeeded" for step in result.trace)
    assert result.trace[0].output_data_object_ids
    assert result.trace[0].output_preview == "SentinelAI is testing real file access."
    assert [
        (edge["parent_id"], edge["child_id"]) for edge in result.provenance_edges
    ] == [
        (
            result.trace[0].input_data_object_ids[0],
            result.trace[0].output_data_object_ids[0],
        ),
        (
            result.trace[0].output_data_object_ids[0],
            result.trace[1].output_data_object_ids[0],
        ),
        (
            result.trace[1].output_data_object_ids[0],
            result.trace[2].output_data_object_ids[0],
        ),
    ]
    assert result.provenance_objects[-1]["transformation"] == "paraphrase"


def test_blocked_demo_stops_and_does_not_invoke_message_tool():
    runtime = create_sentinel_runtime()
    registry = DataObjectRegistry(runtime.provenance_graph)
    source = registry.register_source(
        object_id="blocked-source",
        source="test-fixture:confidential",
        content="synthetic confidential data",
        sensitivity="CONFIDENTIAL",
        created_step="source",
    )
    invocations = []
    runtime.tool_executor._tools["send_message"] = lambda **kwargs: invocations.append(
        kwargs
    )
    loop = DeterministicAgentLoop(runtime, agent_id="blocked-test")

    result = loop.run(
        task_description="Block confidential external transfer.",
        action_plan=[
            PlannedAction(
                step_id="blocked-step",
                tool_name="send_message",
                destination="EXTERNAL",
                description="Attempt simulated external transfer.",
                arguments={"recipient": "demo", "message": "test"},
                input_data_objects=(source.object_id,),
            ),
            PlannedAction(
                step_id="must-not-run",
                tool_name="summarize",
                destination="INTERNAL",
                description="This action must not run after the block.",
                arguments={"text": "not run"},
                input_data_objects=(source.object_id,),
            ),
        ],
        run_id="blocked-run",
    )

    assert result.status == "stopped"
    assert "BLOCK" in result.stop_reason
    assert len(result.trace) == 1
    assert result.trace[0].verdict == "BLOCK"
    assert result.trace[0].execution_status == "not_attempted"
    assert result.trace[0].matched_policy_rules == ["CONFIDENTIAL_TO_EXTERNAL"]
    assert invocations == []
    assert not runtime.provenance_graph.has_object("blocked-step-output")
    assert not runtime.provenance_graph.has_object("must-not-run-output")


def test_warn_stops_without_tool_invocation():
    runtime = create_sentinel_runtime()
    source = DataObjectRegistry(runtime.provenance_graph).register_source(
        object_id="warning-source",
        source="test-fixture:internal",
        content="synthetic internal data",
        sensitivity="INTERNAL",
        created_step="source",
    )
    invocations = []
    runtime.tool_executor._tools["send_message"] = lambda **kwargs: invocations.append(
        kwargs
    )

    result = DeterministicAgentLoop(runtime, "warning-test").run(
        "Stop for a policy warning.",
        [
            PlannedAction(
                step_id="warning-step",
                tool_name="send_message",
                destination="EXTERNAL",
                description="Attempt simulated external message.",
                arguments={"recipient": "demo", "message": "test"},
                input_data_objects=(source.object_id,),
            )
        ],
        run_id="warning-run",
    )

    assert result.status == "stopped"
    assert result.trace[0].verdict == "WARN"
    assert result.trace[0].execution_status == "not_attempted"
    assert invocations == []


def test_every_action_uses_monitor_service_and_outputs_are_reused():
    runtime = create_sentinel_runtime()
    source = DataObjectRegistry(runtime.provenance_graph).register_source(
        object_id="monitor-source",
        source="test-fixture:public",
        content="One sentence for deterministic processing.",
        sensitivity="PUBLIC",
        created_step="source",
    )
    calls = []
    original_process = runtime.monitor_service.process

    def monitored_process(proposal, destination):
        calls.append(proposal.step_id)
        return original_process(proposal, destination)

    runtime.monitor_service.process = monitored_process
    result = DeterministicAgentLoop(runtime, "monitor-test").run(
        "Summarize through the monitor.",
        [
            PlannedAction(
                step_id="monitored-read",
                tool_name="summarize",
                destination="INTERNAL",
                description="Summarize the source.",
                arguments={"text": "One sentence for deterministic processing."},
                input_data_objects=(source.object_id,),
            ),
            PlannedAction(
                step_id="monitored-paraphrase",
                tool_name="paraphrase",
                destination="INTERNAL",
                description="Paraphrase the monitored result.",
                input_from_previous_output=True,
                previous_output_argument="text",
            ),
        ],
        run_id="monitor-run",
    )

    assert result.status == "completed"
    assert calls == ["monitored-read", "monitored-paraphrase"]
    assert result.trace[1].input_data_object_ids == [
        result.trace[0].output_data_object_ids[0]
    ]


def test_tool_failure_is_recorded_and_stops_task():
    runtime = create_sentinel_runtime()
    source = DataObjectRegistry(runtime.provenance_graph).register_source(
        object_id="failure-source",
        source="test-fixture:public",
        content="data",
        sensitivity="PUBLIC",
        created_step="source",
    )
    paraphrase_calls = []

    def fail_paraphrase(**kwargs):
        paraphrase_calls.append(kwargs)
        raise RuntimeError("controlled tool failure")

    runtime.tool_executor._tools["paraphrase"] = fail_paraphrase

    result = DeterministicAgentLoop(runtime, "failure-test").run(
        "Exercise a tool error.",
        [
            PlannedAction(
                step_id="successful-step",
                tool_name="summarize",
                destination="INTERNAL",
                description="A successful first step.",
                arguments={"text": "data"},
                input_data_objects=(source.object_id,),
            ),
            PlannedAction(
                step_id="failure-step",
                tool_name="paraphrase",
                destination="INTERNAL",
                description="A deliberately failing paraphrase invocation.",
                arguments={"text": "data"},
                input_data_objects=(source.object_id,),
            ),
            PlannedAction(
                step_id="after-failure",
                tool_name="paraphrase",
                destination="INTERNAL",
                description="Must not run after an execution failure.",
                arguments={"text": "data"},
                input_data_objects=(source.object_id,),
            ),
        ],
        run_id="failure-run",
    )

    assert result.status == "failed"
    assert len(result.trace) == 2
    assert result.trace[0].execution_status == "succeeded"
    assert result.trace[0].verdict == "ALLOW"
    assert result.trace[1].execution_status == "failed"
    assert "controlled tool failure" in result.trace[1].error
    assert len(paraphrase_calls) == 1


def test_output_registration_failure_is_not_reported_as_tool_success():
    runtime = create_sentinel_runtime()
    source = DataObjectRegistry(runtime.provenance_graph).register_source(
        object_id="output-failure-source",
        source="test-fixture:public",
        content="data",
        sensitivity="PUBLIC",
        created_step="source",
    )
    tool_calls = []
    runtime.tool_executor._tools["summarize"] = lambda **kwargs: (
        tool_calls.append(kwargs) or "summary"
    )

    def fail_output_registration(**kwargs):
        raise ValueError("controlled provenance registration failure")

    runtime.monitor_service.output_handler.create_output_object = (
        fail_output_registration
    )
    result = DeterministicAgentLoop(runtime, "output-failure-test").run(
        "Report post-tool provenance failure accurately.",
        [
            PlannedAction(
                step_id="output-failure-step",
                tool_name="summarize",
                destination="INTERNAL",
                description="Tool succeeds but provenance registration fails.",
                arguments={"text": "data"},
                input_data_objects=(source.object_id,),
            )
        ],
    )

    assert tool_calls == [{"text": "data"}]
    assert result.status == "failed"
    assert result.trace[0].verdict == "ALLOW"
    assert result.trace[0].execution_status == "succeeded"
    assert "controlled provenance registration failure" in result.trace[0].error
    assert result.trace[0].output_data_object_ids == []


@pytest.mark.parametrize(
    "actions",
    [
        [],
        [PlannedAction("bad", "call_external_api", "EXTERNAL", "unsupported")],
        [PlannedAction("bad", "summarize", "INTERNAL", "missing input")],
        [
            PlannedAction(
                "duplicate",
                "summarize",
                "INTERNAL",
                "first",
                input_data_objects=("source",),
            ),
            PlannedAction(
                "duplicate",
                "summarize",
                "INTERNAL",
                "second",
                input_data_objects=("source",),
            ),
        ],
    ],
)
def test_invalid_plans_fail_before_monitoring_or_tool_execution(actions):
    runtime = create_sentinel_runtime()
    source = DataObjectRegistry(runtime.provenance_graph).register_source(
        object_id="validation-source",
        source="test-fixture:public",
        content="data",
        sensitivity="PUBLIC",
        created_step="source",
    )
    actions = [
        PlannedAction(
            action.step_id,
            action.tool_name,
            action.destination,
            action.description,
            arguments=action.arguments,
            input_data_objects=(
                (source.object_id,)
                if action.step_id == "duplicate"
                else action.input_data_objects
            ),
            input_from_previous_output=action.input_from_previous_output,
            previous_output_argument=action.previous_output_argument,
        )
        for action in actions
    ]

    with pytest.raises(ValueError):
        DeterministicAgentLoop(runtime, "invalid-test").run("Invalid plan", actions)

    assert runtime.trajectory_store.count() == 0


def test_unknown_input_reference_is_rejected_before_monitoring():
    runtime = create_sentinel_runtime()
    calls = []
    original_process = runtime.monitor_service.process

    def monitored_process(proposal, destination):
        calls.append(proposal.step_id)
        return original_process(proposal, destination)

    runtime.monitor_service.process = monitored_process
    action = PlannedAction(
        step_id="unknown-input-step",
        tool_name="summarize",
        destination="INTERNAL",
        description="This input reference is invalid.",
        arguments={"text": "data"},
        input_data_objects=("missing-object",),
    )

    with pytest.raises(ValueError, match="missing-object"):
        DeterministicAgentLoop(runtime, "unknown-input-test").run(
            "Reject unknown input refs.",
            [action],
        )

    assert calls == []
    assert runtime.trajectory_store.count() == 0
