from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator

# =====================================================================
# Tool Argument Normalizer Schemas
# =====================================================================


class BaseToolArguments(BaseModel):
    """Base schema for tool arguments that tolerates unexpected keys sent by LLMs."""

    model_config = ConfigDict(
        extra="ignore",  # Silently drop unexpected extra parameters
        populate_by_name=True,  # Allow instantiation via canonical field name or alias
    )


class ReadFileArguments(BaseToolArguments):
    """Normalized arguments for read_file tool."""

    path: str = Field(
        ...,
        validation_alias=AliasChoices(
            "path", "file_path", "filename", "file", "filepath"
        ),
        description="Path or filename of the target file to read",
    )


class WriteMemoryArguments(BaseToolArguments):
    """Normalized arguments for write_memory tool."""

    content: str = Field(
        ...,
        validation_alias=AliasChoices("content", "value", "data", "text", "message"),
    )
    memory_path: Optional[str] = Field(
        None, validation_alias=AliasChoices("memory_path", "path", "file_path", "key")
    )


class SendMessageArguments(BaseToolArguments):
    """Normalized arguments for send_message tool."""

    recipient: str = Field(
        ...,
        validation_alias=AliasChoices(
            "recipient", "to", "receiver", "destination", "target"
        ),
    )
    message: str = Field(
        ..., validation_alias=AliasChoices("message", "content", "body", "text")
    )


class SummarizeArguments(BaseToolArguments):
    """Normalized arguments for summarize tool."""

    text: Optional[str] = Field(
        None, validation_alias=AliasChoices("text", "content", "input_text", "summary")
    )
    max_sentences: int = Field(
        3,
        validation_alias=AliasChoices(
            "max_sentences", "sentences", "limit", "max_length"
        ),
    )


class ParaphraseArguments(BaseToolArguments):
    """Normalized arguments for paraphrase tool."""

    text: Optional[str] = Field(
        None, validation_alias=AliasChoices("text", "content", "input_text")
    )


class CalculateArguments(BaseToolArguments):
    """Normalized arguments for calculate tool."""

    expression: str = Field(
        ...,
        validation_alias=AliasChoices("expression", "expr", "math", "formula", "query"),
    )


class CallExternalApiArguments(BaseToolArguments):
    """Normalized arguments for call_external_api tool."""

    url: str = Field(
        ..., validation_alias=AliasChoices("url", "endpoint", "link", "uri")
    )
    params: Optional[dict] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("params", "parameters", "query_params"),
    )
    timeout: int = Field(10, validation_alias=AliasChoices("timeout", "time_limit"))


# Mapping tool names to their respective argument schema normalizers
TOOL_ARGUMENT_SCHEMAS: dict[str, type[BaseToolArguments]] = {
    "read_file": ReadFileArguments,
    "write_memory": WriteMemoryArguments,
    "read_memory": BaseToolArguments,
    "send_message": SendMessageArguments,
    "summarize": SummarizeArguments,
    "paraphrase": ParaphraseArguments,
    "calculate": CalculateArguments,
    "call_external_api": CallExternalApiArguments,
}


# =====================================================================
# Core System Contracts
# =====================================================================


class ToolCallProposal(BaseModel):
    """
    Represents a tool action proposed by the agent.

    SentinelAI receives this proposal before the tool is executed.
    """

    step_id: str
    agent_id: str

    tool_name: str
    arguments: dict[str, Any]

    input_data_objects: list[str] = Field(default_factory=list)

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def sanitize_and_normalize_arguments(self) -> "ToolCallProposal":
        """
        Normalizes variations in argument key names sent by LLMs into
        canonical parameters expected by underlying tools.
        """
        schema_cls = TOOL_ARGUMENT_SCHEMAS.get(self.tool_name)
        if schema_cls and isinstance(self.arguments, dict):
            try:
                parsed_args = schema_cls.model_validate(self.arguments)
                # Overwrite raw arguments with normalized dictionary
                self.arguments = parsed_args.model_dump(exclude_unset=False)
            except Exception:
                # Fallback to unmodified arguments if normalization fails
                pass
        return self


class Decision(BaseModel):
    """
    Represents SentinelAI's decision for a proposed tool call.
    """

    step_id: str

    verdict: str
    risk_score: float
    risk_probability: Optional[float] = None

    policy_hits: list[str] = Field(default_factory=list)

    provenance_trace: list[str] = Field(default_factory=list)

    reason: str
