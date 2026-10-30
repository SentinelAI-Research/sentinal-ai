from __future__ import annotations

import ast
import math
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ToolArguments(BaseModel):
    model_config = ConfigDict(extra="ignore")


class ReadFileArguments(ToolArguments):
    path: str = Field(default="", description="Path or filename to read")


class SummarizeArguments(ToolArguments):
    max_sentences: int = Field(default=3, ge=1, le=5)


class ParaphraseArguments(ToolArguments):
    pass


class CalculateArguments(ToolArguments):
    expression: str = Field(min_length=1, max_length=128)

    @field_validator("expression")
    @classmethod
    def allow_bounded_arithmetic_only(cls, value: str) -> str:
        try:
            tree = ast.parse(value, mode="eval")
        except SyntaxError as exc:
            raise ValueError("Expression is not valid arithmetic.") from exc
        nodes = list(ast.walk(tree))
        if len(nodes) > 64:
            raise ValueError("Expression is too complex.")
        allowed = (
            ast.Expression,
            ast.BinOp,
            ast.UnaryOp,
            ast.Constant,
            ast.Add,
            ast.Sub,
            ast.Mult,
            ast.Div,
            ast.UAdd,
            ast.USub,
            ast.Load,
        )
        for node in nodes:
            if not isinstance(node, allowed):
                raise ValueError("Only bounded basic arithmetic is allowed.")
            if isinstance(node, ast.Constant):
                if isinstance(node.value, bool) or not isinstance(
                    node.value, (int, float)
                ):
                    raise ValueError("Expression constants must be numbers.")
                if not math.isfinite(float(node.value)) or abs(node.value) > 1_000_000:
                    raise ValueError(
                        "Expression constants are outside the allowed range."
                    )
        return value


class WriteMemoryArguments(ToolArguments):
    pass


class ReadMemoryArguments(ToolArguments):
    pass


class SendMessageArguments(ToolArguments):
    recipient: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=500)


TOOL_ARGUMENT_MODELS: dict[str, type[ToolArguments]] = {
    "read_file": ReadFileArguments,
    "summarize": SummarizeArguments,
    "paraphrase": ParaphraseArguments,
    "calculate": CalculateArguments,
    "write_memory": WriteMemoryArguments,
    "read_memory": ReadMemoryArguments,
    "send_message": SendMessageArguments,
}

TOOL_DESTINATIONS = {
    "read_file": "INTERNAL",
    "summarize": "INTERNAL",
    "paraphrase": "INTERNAL",
    "calculate": "INTERNAL",
    "write_memory": "MEMORY",
    "read_memory": "MEMORY",
    "send_message": "EXTERNAL",
}

TOOL_DESCRIPTIONS = {
    "read_file": "Read only the designated file copied into this run's safe workspace.",
    "summarize": "Summarize the most recent authorized tool output.",
    "paraphrase": "Paraphrase the most recent authorized tool output.",
    "calculate": "Evaluate a bounded arithmetic expression; no code or network access.",
    "write_memory": "Write the most recent authorized tool output to this run's isolated memory.",
    "read_memory": "Read this run's isolated memory.",
    "send_message": "Simulate an outbound message. No real message is sent.",
}

PROVIDER_PARAMETER_SCHEMAS = {
    "read_file": {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path or filename to read"}
        },
        "required": [],
        "additionalProperties": True,
    },
    "summarize": {
        "type": "object",
        "properties": {"max_sentences": {"type": "integer"}},
        "required": ["max_sentences"],
        "additionalProperties": True,
    },
    "paraphrase": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True,
    },
    "calculate": {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
        "additionalProperties": True,
    },
    "write_memory": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True,
    },
    "read_memory": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": True,
    },
    "send_message": {
        "type": "object",
        "properties": {
            "recipient": {"type": "string"},
            "message": {"type": "string"},
        },
        "required": ["recipient", "message"],
        "additionalProperties": True,
    },
}

SENSITIVE_ARGUMENTS = {"message", "text", "content", "path", "recipient"}


def tool_schemas() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": name,
            "description": TOOL_DESCRIPTIONS[name],
            "parameters": PROVIDER_PARAMETER_SCHEMAS[name],
        }
        for name, model in TOOL_ARGUMENT_MODELS.items()
    )


def validate_tool_arguments(tool_name: str, arguments: Any) -> dict[str, Any]:
    model = TOOL_ARGUMENT_MODELS.get(tool_name)
    if model is None:
        raise ValueError(f"Unsupported tool: {tool_name}")
    if not isinstance(arguments, dict):
        arguments = {}
    return model.model_validate(arguments).model_dump()


def redact_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    redacted: dict[str, Any] = {}
    for key, value in arguments.items():
        if key in SENSITIVE_ARGUMENTS and isinstance(value, str):
            redacted[key] = f"[redacted {len(value)} characters]"
        else:
            redacted[key] = value
    return redacted
