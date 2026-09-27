from typing import Callable

"""
ToolExecutor
    ↓
registered tools only

This gives us controlled execution boundary
"""
class ToolExecutor:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register_tool(
        self,
        name: str,
        function: Callable,
    ) -> None:
        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = function

    def execute(
        self,
        tool_name: str,
        arguments: dict,
    ):
        if tool_name not in self._tools:
            raise ValueError(f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]

        return tool(**arguments)
