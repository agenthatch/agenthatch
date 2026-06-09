"""Memory recall tool — builtin tool for MemoryBrick (v0.7.6).

The `recall` tool enables the LLM to actively query its own memory
for relevant past interactions and knowledge. Registered as a CapBus
builtin tool when MemoryBrick is active.
"""

from __future__ import annotations

from typing import Any


class RecallTool:
    """Builtin tool that lets the LLM search agent memory.

    Registered on CapBus when MemoryBrick is assembled. The LLM calls
    `recall(query, limit)` to search past sessions, knowledge facts,
    and core memory for relevant information.
    """

    description = (
        "Search the agent's memory for relevant past interactions and "
        "knowledge. Use this when you need context from prior conversations "
        "or stored facts."
    )

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "What to search for (natural language).",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results (1-10, default 5).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def __init__(self, memory_brick: Any = None):
        """Initialize with a reference to the MemoryBrick instance.

        Args:
            memory_brick: The MemoryBrick instance to search.
        """
        self._memory = memory_brick

    def execute(self, query: str, limit: int = 5) -> str:
        """Execute a memory recall search.

        Args:
            query: Natural language search query.
            limit: Maximum number of results (clamped to 1-10).

        Returns:
            Formatted search results as a string.
        """
        if self._memory is None:
            return "Memory is not available."

        return self._memory.recall(query, limit=min(max(limit, 1), 10))

    @property
    def tool_definition(self) -> Any:
        """Return the tool definition for CapBus registration."""
        from agenthatch_core.tools.bus import ToolDefinition
        return ToolDefinition(
            function={
                "name": "recall",
                "description": self.description,
                "parameters": self.schema,
            }
        )


def recall_tool(memory_brick: Any = None) -> RecallTool:
    """Factory function for creating a RecallTool instance.

    Args:
        memory_brick: The MemoryBrick instance.

    Returns:
        A RecallTool instance ready for CapBus registration.
    """
    return RecallTool(memory_brick)