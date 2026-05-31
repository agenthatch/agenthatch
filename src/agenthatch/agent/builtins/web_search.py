"""Web search builtin capability."""

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class WebSearchCap(BuiltinCapability):
    name = "web_search"
    cap_type = "reasoning"
    description = "Search the web for information"
    schema = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query"},
        },
        "required": ["query"],
    }

    def execute(self, query: str = "") -> str:  # type: ignore[override]
        return (
            f"Web search for '{query}' is not yet implemented. "
            "This capability requires an external search API key."
        )


BUILTIN_REGISTRY["web_search"] = WebSearchCap
