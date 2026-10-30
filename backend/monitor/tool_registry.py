from pathlib import Path

from backend.monitor.executor import ToolExecutor

from backend.tools.communication_tools import send_message
from backend.tools.file_tools import (
    calculate,
    paraphrase,
    read_file,
    summarize,
)
from backend.tools.memory_tools import (
    read_memory,
    write_memory,
)


def create_tool_executor(
    file_root: str | Path | None = None,
    memory_path: str | Path | None = None,
) -> ToolExecutor:
    executor = ToolExecutor()

    if file_root is None:
        registered_read_file = read_file
    else:
        allowed_root = Path(file_root).resolve()

        def registered_read_file(path: str) -> str:
            resolved_path = Path(path).resolve()
            if not resolved_path.is_relative_to(allowed_root):
                raise ValueError("File path is outside the run workspace.")
            return read_file(str(resolved_path))

    if memory_path is None:
        registered_write_memory = write_memory
        registered_read_memory = read_memory
    else:
        isolated_memory_path = Path(memory_path).resolve()

        def registered_write_memory(content: str) -> str:
            return write_memory(content, memory_path=isolated_memory_path)

        def registered_read_memory() -> str:
            return read_memory(memory_path=isolated_memory_path)

    executor.register_tool(
        "read_file",
        registered_read_file,
    )

    executor.register_tool(
        "summarize",
        summarize,
    )

    executor.register_tool(
        "paraphrase",
        paraphrase,
    )

    executor.register_tool(
        "calculate",
        calculate,
    )

    executor.register_tool(
        "write_memory",
        registered_write_memory,
    )

    executor.register_tool(
        "read_memory",
        registered_read_memory,
    )

    executor.register_tool(
        "send_message",
        send_message,
    )

    return executor
