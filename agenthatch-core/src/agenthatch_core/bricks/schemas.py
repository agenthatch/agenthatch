"""OutputSchema — compile Pydantic response models from schema dicts.

Level 0 — converts declarative JSON Schema definitions into runtime
Pydantic models using create_model().  Used by the loop engine to
enforce structured output for skills that declare an output_schema.
"""

from __future__ import annotations

from typing import Any


def compile_output_schema(
    schema: dict[str, Any],
    model_name: str = "OutputModel",
) -> type:
    """Compile a JSON Schema dict into a Pydantic model.

    Args:
        schema: JSON Schema dict with "properties" and optional "required".
        model_name: Name for the generated Pydantic model.

    Returns:
        A Pydantic BaseModel subclass.

    Example:
        schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "score": {"type": "integer"},
            },
            "required": ["summary", "score"],
        }
        Model = compile_output_schema(schema, "AnalysisOutput")
    """
    from pydantic import BaseModel, Field, create_model

    properties = schema.get("properties", {})
    required: set[str] = set(schema.get("required", []))

    type_map: dict[str, type] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }

    fields: dict[str, tuple[type, Any]] = {}
    for name, prop in properties.items():
        json_type = prop.get("type", "string")
        py_type = type_map.get(json_type, str)
        description = prop.get("description", "")

        if name in required:
            fields[name] = (py_type, Field(description=description))
        else:
            fields[name] = (
                py_type | None,
                Field(default=None, description=description),
            )

    # Ensure at least one field — pydantic create_model requires fields
    if not fields:
        fields["result"] = (str, Field(default="", description="Output text"))

    return create_model(model_name, **fields, __base__=BaseModel)  # type: ignore[call-overload]