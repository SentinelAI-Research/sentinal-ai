from types import SimpleNamespace

import pytest

from backend.agent.llm_providers import (
    OpenAIChatProvider,
    ProviderError,
    ProviderRequest,
    _function_tools,
)
from backend.agent.tool_contracts import tool_schemas


class FakeCompletions:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.response


def make_response(function_name, arguments, request_id="request-test", usage=None):
    tool_call = SimpleNamespace(
        function=SimpleNamespace(name=function_name, arguments=arguments)
    )
    message = SimpleNamespace(tool_calls=[tool_call])
    choice = SimpleNamespace(message=message)
    return SimpleNamespace(
        choices=[choice],
        id=request_id,
        usage=usage,
    )


def test_provider_strict_schemas_exclude_trusted_security_fields():
    tools = _function_tools(tool_schemas())
    names = {tool["function"]["name"] for tool in tools}
    assert "propose_read_file" in names
    assert "propose_send_message" in names
    assert "finish_task" in names
    for tool in tools:
        parameters = tool["function"]["parameters"]
        assert parameters.get("additionalProperties") is False
        serialized = str(parameters)
        assert "destination" not in serialized
        assert "sensitivity" not in serialized
        assert "verdict" not in serialized
        assert "input_data_object_ids" not in serialized


def test_provider_parses_one_structured_action_and_real_usage():
    response = make_response(
        "propose_summarize",
        '{"max_sentences":3}',
        usage=SimpleNamespace(prompt_tokens=21, completion_tokens=8, total_tokens=29),
    )
    provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
    provider.model_name = "test-model"
    completions = FakeCompletions(response=response)
    provider._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    request = ProviderRequest(
        task_description="Summarize public material",
        messages=[{"role": "user", "content": "Summarize this."}],
        tool_schemas=tool_schemas(),
        action_index=1,
        timeout_seconds=4,
    )

    turn = provider.next_turn(request)

    assert turn.kind == "action"
    assert turn.tool_name == "summarize"
    assert turn.arguments == {"max_sentences": 3}
    assert turn.request_id == "request-test"
    assert turn.usage == {"input_tokens": 21, "output_tokens": 8, "total_tokens": 29}
    assert completions.kwargs["parallel_tool_calls"] is False
    assert completions.kwargs["timeout"] == 4


def test_provider_rejects_malformed_or_multiple_tool_calls():
    provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
    provider.model_name = "test-model"
    provider._client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=FakeCompletions(
                response=SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=[]))],
                    id="request-empty",
                    usage=None,
                )
            )
        )
    )
    request = ProviderRequest(
        task_description="test",
        messages=[],
        tool_schemas=tool_schemas(),
        action_index=1,
        timeout_seconds=3,
    )

    with pytest.raises(ProviderError, match="structured action"):
        provider.next_turn(request)


def test_provider_rejects_malformed_arguments_and_unknown_functions():
    provider = OpenAIChatProvider.__new__(OpenAIChatProvider)
    provider.model_name = "test-model"
    request = ProviderRequest(
        task_description="test",
        messages=[],
        tool_schemas=tool_schemas(),
        action_index=1,
        timeout_seconds=3,
    )

    for response in (
        make_response("propose_summarize", "not-json"),
        make_response("propose_unknown", "{}"),
    ):
        provider._client = SimpleNamespace(
            chat=SimpleNamespace(completions=FakeCompletions(response=response))
        )
        with pytest.raises(ProviderError, match="structured action"):
            provider.next_turn(request)
