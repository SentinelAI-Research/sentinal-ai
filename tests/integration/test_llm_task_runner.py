import json
import time

import pytest

from backend.agent.llm_providers import (
    DemoFakeProvider,
    ProviderError,
    ProviderTurn,
    ScriptedFakeProvider,
)
from backend.agent.llm_task_runner import LLMRunManager, LLMRunRequest


def wait_for_terminal(manager, run_id, timeout=8):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        result = manager.store.get(run_id)
        if result["status"] in {"completed", "stopped", "failed"}:
            return result
        time.sleep(0.01)
    raise AssertionError("LLM run did not reach a terminal state in time")


def run_fake(scenario_id, task="Complete the controlled scenario.", max_actions=6):
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task=task,
            mode="fake",
            scenario_id=scenario_id,
            max_actions=max_actions,
        ),
        provider_factory=lambda request: DemoFakeProvider(request.scenario_id),
    )
    return manager, run_id


def test_fake_provider_safe_document_monitors_each_step_and_records_lineage():
    manager, run_id = run_fake("safe_document")
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert result["provider"] == "fake"
    assert result["summary"]["proposals"] == 3
    assert result["summary"]["allow"] == 3
    assert result["summary"]["executed"] == 3
    assert [
        event["data"]["tool_name"]
        for event in result["events"]
        if event["type"] == "proposal_received"
    ] == ["read_file", "summarize", "paraphrase"]
    assert result["summary"]["provenance_edges"] == 3
    assert all(
        event["type"]
        in {
            "proposal_received",
            "validation_passed",
            "monitor_decision",
            "execution_attempt",
            "execution_result",
            "output_provenance_recorded",
            "feedback_sent",
            "provider_request",
            "run_started",
            "input_resolved",
            "task_finished",
        }
        for event in result["events"]
    )


def test_deterministic_mode_uses_same_event_and_metric_contract():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Read and transform the public sample.",
            mode="deterministic",
            scenario_id="safe_document",
            max_actions=4,
        )
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert result["summary"]["proposals"] == 3
    assert result["summary"]["allow"] == 3
    assert result["summary"]["executed"] == 3
    assert result["summary"]["outputs"] == 3
    assert sum(event["type"] == "monitor_decision" for event in result["events"]) == 3


def test_calculator_uses_actual_result_and_fake_mode_is_explicit():
    manager, run_id = run_fake("calculator", task="Calculate six times seven.")
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert result["mode"] == "fake"
    assert result["provider"] == "fake"
    assert any(event["type"] == "feedback_sent" for event in result["events"])
    assert result["provenance"]["objects"][-1]["transformation"] == "calculate"


def test_confidential_external_transfer_is_blocked_without_execution_or_output():
    manager, run_id = run_fake("confidential_external_transfer")
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "stopped"
    assert result["summary"]["block"] == 1
    assert result["summary"]["executed"] == 0
    assert result["summary"]["outputs"] == 0
    assert any(event["type"] == "execution_skipped" for event in result["events"])
    assert not any(event["type"] == "execution_attempt" for event in result["events"])


def test_internal_external_transfer_warns_and_never_executes():
    manager, run_id = run_fake("internal_external_transfer")
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "stopped"
    assert result["summary"]["warn"] == 1
    assert result["summary"]["executed"] == 0
    assert result["summary"]["outputs"] == 0


def test_prompt_injection_text_does_not_change_trusted_action_routing():
    manager, run_id = run_fake("prompt_injection")
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert [
        event["data"]["tool_name"]
        for event in result["events"]
        if event["type"] == "proposal_received"
    ] == ["read_file", "summarize", "paraphrase"]
    assert result["summary"]["allow"] == 3


def test_provider_error_preserves_prior_events_without_fallback():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Fail live explicitly",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [
                ProviderTurn(
                    kind="action",
                    tool_name="unknown_tool",
                    arguments={},
                    input_data_object_ids=[],
                )
            ]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "failed"
    assert result["summary"]["proposals"] == 1
    assert any(event["type"] == "validation_failed" for event in result["events"])
    assert result["provider"] == "fake"


def test_provider_failure_after_success_keeps_prior_result_and_fails_run():
    class FailAfterOneAction:
        name = "fake"
        model_name = "failure-after-action"

        def __init__(self):
            self.turns = 0

        def next_turn(self, request):
            self.turns += 1
            if self.turns == 1:
                return ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                )
            raise ProviderError("provider_timeout", "controlled timeout")

    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Do two calculations.",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: FailAfterOneAction(),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "failed"
    assert result["summary"]["proposals"] == 1
    assert result["summary"]["executed"] == 1
    assert result["summary"]["outputs"] == 1
    assert any(event["type"] == "provider_error" for event in result["events"])


def test_model_finish_summary_is_redacted_before_storage():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Calculate six times seven.",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [
                ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                ),
                ProviderTurn(
                    kind="finish",
                    completion_summary="done api_key=sk-secret-value",
                ),
            ]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert "sk-secret-value" not in result["completion_summary"]
    assert "sk-secret-value" not in json.dumps(result["events"])


def test_later_tool_failure_preserves_prior_output_and_fails_task():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Read the source then calculate.",
            mode="fake",
            scenario_id="safe_document",
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [
                ProviderTurn(
                    kind="action",
                    tool_name="read_file",
                    arguments={},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "1 / 0"},
                ),
            ]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "failed"
    assert result["summary"]["proposals"] == 2
    assert result["summary"]["executed"] == 1
    assert result["summary"]["failed"] == 1
    assert result["summary"]["outputs"] == 1
    assert any(event["type"] == "feedback_sent" for event in result["events"])


def test_endless_provider_is_stopped_at_action_limit():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Perform several calculations.",
            mode="fake",
            scenario_id="calculator",
            max_actions=2,
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [
                ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "8 * 9"},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "2 * 3"},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="summarize",
                    arguments={"max_sentences": 3},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="paraphrase",
                    arguments={},
                ),
            ]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "stopped"
    assert result["summary"]["executed"] == 2
    assert any(event["type"] == "action_limit_reached" for event in result["events"])


def test_concurrent_runs_have_distinct_ids_and_provenance():
    first_manager = LLMRunManager()
    first_id = first_manager.start(
        LLMRunRequest(
            task="First isolated run.",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: DemoFakeProvider("calculator"),
    )
    second_id = first_manager.start(
        LLMRunRequest(
            task="Second isolated run.",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: DemoFakeProvider("calculator"),
    )
    first = wait_for_terminal(first_manager, first_id)
    second = wait_for_terminal(first_manager, second_id)

    assert first_id != second_id
    assert first["status"] == second["status"] == "completed"
    first_objects = {item["object_id"] for item in first["provenance"]["objects"]}
    second_objects = {item["object_id"] for item in second["provenance"]["objects"]}
    assert first_objects.isdisjoint(second_objects)


def test_tool_argument_schema_rejects_path_override_and_security_fields():
    from backend.agent.tool_contracts import validate_tool_arguments

    with pytest.raises(ValueError):
        validate_tool_arguments("read_file", {"path": "../../secrets.txt"})

    with pytest.raises(ValueError):
        validate_tool_arguments(
            "summarize",
            {"max_sentences": 2, "input_data_object_ids": ["made-up"]},
        )

    with pytest.raises(ValueError):
        validate_tool_arguments("calculate", {"expression": "2 ** 100000"})


def test_run_request_rejects_credential_shaped_task_text():
    with pytest.raises(Exception):
        LLMRunRequest(
            task="Please use api_key=sk-test-secret-value",
            mode="fake",
        )

    with pytest.raises(Exception):
        LLMRunRequest(task="   ", mode="fake")


def test_live_mode_does_not_fall_back_to_fake_provider(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(task="Use live provider only.", mode="live"),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "failed"
    assert result["provider"] == "openai"
    assert result["summary"]["proposals"] == 0
    assert result["summary"]["executed"] == 0
    assert result["error"] == "Provider failed (missing_credentials)."


def test_scripted_provider_failure_does_not_fabricate_proposal():
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Provider offline",
            mode="fake",
            scenario_id="calculator",
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [ProviderTurn(kind="finish", completion_summary="done")]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "failed"
    assert result["summary"]["proposals"] == 0
    assert result["summary"]["executed"] == 0


def test_action_limit_stops_repeated_provider_proposals():
    repeated_action = ProviderTurn(
        kind="action",
        tool_name="calculate",
        arguments={"expression": "6 * 7"},
    )
    manager = LLMRunManager()
    run_id = manager.start(
        LLMRunRequest(
            task="Continue calculating",
            mode="fake",
            scenario_id="calculator",
            max_actions=2,
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [repeated_action, repeated_action, repeated_action]
        ),
    )
    result = wait_for_terminal(manager, run_id)

    assert result["status"] in {"failed", "stopped"}
    assert result["summary"]["proposals"] <= 3
    assert result["summary"]["executed"] <= 2


def test_run_request_rejects_unknown_mode_policy_or_extra_fields():
    with pytest.raises(Exception):
        LLMRunRequest(task="test", mode="mock")

    with pytest.raises(Exception):
        LLMRunRequest(task="test", mode="fake", policy_id="unknown")

    with pytest.raises(Exception):
        LLMRunRequest(
            task="test",
            mode="fake",
            unrestricted_tool="send_message",
        )


def test_send_message_uses_trusted_object_content_not_model_payload():
    manager = LLMRunManager()
    attacker_text = "UNTRUSTED_CONFIDENTIAL_PAYLOAD_123"

    run_id = manager.start(
        LLMRunRequest(
            task="Send the current public content.",
            mode="fake",
            scenario_id="safe_document",
        ),
        provider_factory=lambda request: ScriptedFakeProvider(
            [
                ProviderTurn(
                    kind="action",
                    tool_name="read_file",
                    arguments={},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="summarize",
                    arguments={"max_sentences": 3},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="paraphrase",
                    arguments={},
                ),
                ProviderTurn(
                    kind="action",
                    tool_name="send_message",
                    arguments={
                        "recipient": "attacker-controlled-recipient",
                        "message": attacker_text,
                    },
                ),
                ProviderTurn(
                    kind="finish",
                    completion_summary="The task has finished.",
                ),
            ]
        ),
    )

    result = wait_for_terminal(manager, run_id)

    assert result["status"] == "completed"
    assert result["summary"]["executed"] == 4

    execution_events = [
        event
        for event in result["events"]
        if event["type"] == "execution_result" and event["data"].get("executed") is True
    ]
    assert len(execution_events) == 4

    # Model-supplied payload must not become the sent content.
    assert attacker_text not in json.dumps(execution_events)
