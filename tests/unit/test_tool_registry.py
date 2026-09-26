from backend.monitor.tool_registry import (
    create_tool_executor,
)


def test_default_tool_registry():
    executor = create_tool_executor()

    result = executor.execute(
        "calculate",
        {
            "expression": "2 + 3",
        },
    )

    assert result == 5
