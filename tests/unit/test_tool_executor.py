from backend.monitor.executor import ToolExecutor
import pytest


def test_tool_can_be_registered_and_executed():
    executor = ToolExecutor()

    def add_numbers(a: int, b: int):
        return a + b

    executor.register_tool(
        "add",
        add_numbers,
    )

    result = executor.execute(
        "add",
        {
            "a": 2,
            "b": 3,
        },
    )

    assert result == 5


def test_unknown_tool_is_rejected():
    executor = ToolExecutor()

    with pytest.raises(ValueError):
        executor.execute(
            "unknown_tool",
            {},
        )
