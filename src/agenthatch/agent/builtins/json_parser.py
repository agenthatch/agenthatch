"""JSON parser builtin capability."""

import json

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class JsonParserCap(BuiltinCapability):
    name = "json_parser"
    cap_type = "utility"
    description = "Parse and query JSON data"
    schema = {
        "type": "object",
        "properties": {
            "json_string": {"type": "string", "description": "JSON string to parse"},
            "query": {"type": "string", "description": "JSONPath-like query key"},
        },
        "required": ["json_string"],
    }

    def execute(self, json_string: str = "", query: str = "") -> str:  # type: ignore[override]
        try:
            data = json.loads(json_string)
        except json.JSONDecodeError as e:
            return f"Error parsing JSON: {e}"
        if query:
            keys = query.split(".")
            for key in keys:
                if isinstance(data, dict):
                    data = data.get(key)
                elif isinstance(data, list):
                    try:
                        data = data[int(key)]
                    except (IndexError, ValueError):
                        return f"Error: index '{key}' out of range"
                else:
                    return f"Error: cannot query key '{key}' on {type(data).__name__}"
            if data is None:
                return "null"
        return json.dumps(data, indent=2, ensure_ascii=False)


BUILTIN_REGISTRY["json_parser"] = JsonParserCap
