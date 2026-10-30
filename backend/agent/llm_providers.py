from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ProviderRequest:
    task_description: str
    messages: list[dict[str, str]]
    tool_schemas: tuple[dict[str, Any], ...]
    action_index: int
    timeout_seconds: float


@dataclass(frozen=True)
class ProviderTurn:
    kind: Literal["action", "finish"]
    tool_name: str | None = None
    arguments: Any = None
    input_data_object_ids: Any = None
    explanation: str = ""
    completion_summary: str = ""
    request_id: str | None = None
    usage: dict[str, int] | None = None


class ProviderError(RuntimeError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def redact_untrusted_text(text: str, maximum_length: int = 300) -> str:
    import re

    redacted = re.sub(
        r"(?i)\b(api[_ -]?key|access[_ -]?token|password)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        text,
    )
    redacted = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [redacted]", redacted)
    redacted = re.sub(r"\bsk-[A-Za-z0-9_-]{20,}", "[redacted-key]", redacted)
    redacted = re.sub(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        "[redacted-private-key]",
        redacted,
        flags=re.DOTALL,
    )
    return redacted[:maximum_length]


class LLMProvider(Protocol):
    name: str
    model_name: str

    def next_turn(
        self,
        request: ProviderRequest,
    ) -> ProviderTurn:
        """Return one structured proposal or an explicit finish response."""


def _function_tools(
    tool_schemas: tuple[dict[str, Any], ...],
) -> list[dict[str, Any]]:
    tools = []

    for schema in tool_schemas:
        parameters = dict(schema["parameters"])
        properties = parameters.get("properties") or {}

        if not properties:
            parameters = {
                "type": "object",
                "properties": {},
                "additionalProperties": True,
            }

        tools.append(
            {
                "type": "function",
                "function": {
                    "name": f"propose_{schema['name']}",
                    "description": (
                        schema["description"] + " This proposal is not authorization. "
                        "SentinelAI independently validates and decides."
                    ),
                    "parameters": parameters,
                    "strict": False,
                },
            }
        )

    tools.append(
        {
            "type": "function",
            "function": {
                "name": "finish_task",
                "description": "Finish when the requested task is complete.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "summary": {"type": "string"},
                    },
                    "required": ["summary"],
                    "additionalProperties": False,
                },
                "strict": True,
            },
        }
    )

    return tools


class OpenAIChatProvider:
    """OpenAI official SDK adapter using structured function calling with rate-limit retry support."""

    name = "openai"

    def __init__(self):
        self.model_name = os.getenv(
            "SENTINELAI_GROQ_MODEL",
            "openai/gpt-oss-20b",
        )
        credential = os.getenv("GROQ_API_KEY")
        if not credential:
            raise ProviderError(
                "missing_credentials",
                "GROQ_API_KEY is not configured for live mode.",
            )
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ProviderError(
                "sdk_unavailable",
                "Install the OpenAI SDK from requirements.txt to use live mode.",
            ) from exc
        try:
            self._client = OpenAI(
                api_key=credential,
                base_url="https://api.groq.com/openai/v1",
                max_retries=0,
            )
        except Exception as exc:
            raise ProviderError(
                "provider_configuration_failed",
                f"OpenAI client configuration failed ({type(exc).__name__}).",
            ) from None

    def next_turn(self, request: ProviderRequest) -> ProviderTurn:
        max_retries = 3
        backoff = 3.0

        for attempt in range(max_retries + 1):
            try:
                response = self._client.chat.completions.create(
                    model=self.model_name,
                    messages=request.messages,
                    tools=_function_tools(request.tool_schemas),
                    tool_choice="auto",
                    parallel_tool_calls=False,
                    timeout=request.timeout_seconds,
                    max_completion_tokens=500,
                )
                break
            except Exception as exc:
                exception_name = type(exc).__name__
                status_code = getattr(exc, "status_code", None)
                error_code = getattr(exc, "code", None)
                error_message = getattr(exc, "message", None) or str(exc)

                print(
                    "[SentinelAI OpenAI diagnostic] "
                    f"type={exception_name}, "
                    f"status={status_code}, "
                    f"code={error_code}, "
                    f"message={error_message}"
                )

                if exception_name == "RateLimitError" or status_code == 429:
                    if attempt < max_retries:
                        print(
                            f"[SentinelAI Rate Limit] Retrying in {backoff} seconds (Attempt {attempt + 1}/{max_retries})..."
                        )
                        time.sleep(backoff)
                        backoff *= 2
                        continue
                    raise ProviderError(
                        "provider_rate_limited", "Groq rate limit exceeded."
                    ) from None

                if exception_name == "APITimeoutError":
                    code = "provider_timeout"
                elif exception_name in {
                    "AuthenticationError",
                    "PermissionDeniedError",
                }:
                    code = "provider_authentication_failed"
                elif exception_name == "APIConnectionError":
                    code = "provider_unavailable"
                else:
                    code = "provider_request_failed"

                raise ProviderError(
                    code, f"Provider request failed ({exception_name})."
                ) from None

        try:
            choice = response.choices[0]
            calls = choice.message.tool_calls or []

            usage = None
            if response.usage is not None:
                usage = {
                    name: value
                    for name, value in (
                        ("input_tokens", response.usage.prompt_tokens),
                        ("output_tokens", response.usage.completion_tokens),
                        ("total_tokens", response.usage.total_tokens),
                    )
                    if value is not None
                }
            request_id = getattr(response, "id", None)

            if not calls:
                content = choice.message.content or "Task completed."
                return ProviderTurn(
                    kind="finish",
                    completion_summary=content.strip(),
                    request_id=request_id,
                    usage=usage,
                )

            if len(calls) != 1:
                raise ValueError("Expected exactly one structured function call.")

            call = calls[0]
            arguments = json.loads(call.function.arguments or "{}")
            if not isinstance(arguments, dict):
                arguments = {}

            if call.function.name == "finish_task":
                summary = arguments.get("summary")
                if not isinstance(summary, str):
                    summary = str(summary) if summary is not None else "Task completed."
                return ProviderTurn(
                    kind="finish",
                    completion_summary=summary,
                    request_id=request_id,
                    usage=usage,
                )

            if not call.function.name.startswith("propose_"):
                raise ValueError(
                    f"Unknown provider function name: {call.function.name}"
                )

            proposed_tool_name = call.function.name.removeprefix("propose_")
            allowed_tool_names = {schema["name"] for schema in request.tool_schemas}
            if proposed_tool_name not in allowed_tool_names:
                raise ValueError(
                    "Provider proposed a tool not included in its request."
                )

            return ProviderTurn(
                kind="action",
                tool_name=proposed_tool_name,
                arguments=arguments,
                explanation="Model proposed one structured tool call.",
                request_id=request_id,
                usage=usage,
            )
        except Exception as exc:
            print(f"[SentinelAI Provider Parse Error]: {exc}")
            raise ProviderError(
                "malformed_provider_response",
                f"Provider did not return a valid structured action or finish call: {exc}",
            ) from None


class ScriptedFakeProvider:
    """Offline provider for tests. Exhaustion is an error, never a fallback."""

    name = "fake"
    model_name = "sentinelai-scripted-fake-v1"

    def __init__(self, turns: list[ProviderTurn]):
        self._turns = list(turns)
        self.request_count = 0

    def next_turn(self, request: ProviderRequest) -> ProviderTurn:
        self.request_count += 1
        if not self._turns:
            raise ProviderError(
                "fake_script_exhausted",
                "The explicit fake-provider script has no next response.",
            )
        return self._turns.pop(0)


class DemoFakeProvider:
    """Explicit fake mode demonstrating the same safe public-file workflow."""

    name = "fake"
    model_name = "sentinelai-demo-fake-v1"

    def __init__(self, scenario_id: str = "safe_document"):
        self.scenario_id = scenario_id
        self._index = 0

    def next_turn(self, request: ProviderRequest) -> ProviderTurn:
        index = self._index
        self._index += 1
        if self.scenario_id in {"safe_document", "prompt_injection"}:
            if index == 0:
                tool_name = "read_file"
                arguments: dict[str, Any] = {"path": "data/sample/sample_document.txt"}
                explanation = "Read the designated run-local document."
            elif index == 1:
                tool_name = "summarize"
                arguments = {"max_sentences": 3}
                explanation = "Summarize the public document result."
            elif index == 2:
                tool_name = "paraphrase"
                arguments = {}
                explanation = "Create a derived draft from the summary."
            else:
                return ProviderTurn(
                    kind="finish",
                    completion_summary="The fake-provider document task completed.",
                )
            return ProviderTurn(
                kind="action",
                tool_name=tool_name,
                arguments=arguments,
                explanation=explanation,
            )
        if self.scenario_id == "calculator":
            if index == 0:
                return ProviderTurn(
                    kind="action",
                    tool_name="calculate",
                    arguments={"expression": "6 * 7"},
                    explanation="Calculate the requested safe expression.",
                )
            return ProviderTurn(
                kind="finish",
                completion_summary="The calculator task completed.",
            )
        if self.scenario_id == "confidential_external_transfer":
            return ProviderTurn(
                kind="action",
                tool_name="send_message",
                arguments={
                    "recipient": "demo-recipient",
                    "message": "Synthetic confidential transfer.",
                },
                explanation="Attempt the controlled external-transfer scenario.",
            )
        if self.scenario_id == "internal_external_transfer":
            return ProviderTurn(
                kind="action",
                tool_name="send_message",
                arguments={
                    "recipient": "demo-recipient",
                    "message": "Synthetic internal transfer.",
                },
                explanation="Attempt the controlled warning scenario.",
            )
        raise ProviderError(
            "unknown_fake_scenario", "Fake provider scenario is unknown."
        )
