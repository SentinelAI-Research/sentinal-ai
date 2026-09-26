from pydantic import BaseModel


class ExecutionResult(BaseModel):
    executed: bool
    tool_name: str
    output: str | None = None
    error: str | None = None
    output_data_object_id: str | None = None
