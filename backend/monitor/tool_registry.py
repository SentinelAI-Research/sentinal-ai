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


def create_tool_executor() -> ToolExecutor:
    executor = ToolExecutor()

    executor.register_tool(
        "read_file",
        read_file,
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
        write_memory,
    )

    executor.register_tool(
        "read_memory",
        read_memory,
    )

    executor.register_tool(
        "send_message",
        send_message,
    )

    return executor
