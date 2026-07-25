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

    _TYPE_MAP: dict[str, type] = {
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
        py_type = _TYPE_MAP.get(json_type, str)
        description = prop.get("description", "")
        field_kwargs: dict[str, Any] = {}

        if description:
            field_kwargs["description"] = description

        # ── enum → Literal ──
        enum_values = prop.get("enum")
        if enum_values is not None:
            from typing import Literal
            py_type = Literal[tuple(enum_values)]  # type: ignore[valid-type,misc]

        # ── Numeric constraints: minimum / maximum ──
        if json_type in ("integer", "number"):
            if "minimum" in prop:
                field_kwargs["ge"] = prop["minimum"]
            if "maximum" in prop:
                field_kwargs["le"] = prop["maximum"]

        # ── String constraints: minLength / maxLength / pattern ──
        if json_type == "string":
            str_constraints: dict[str, Any] = {}
            if "minLength" in prop:
                str_constraints["min_length"] = prop["minLength"]
            if "maxLength" in prop:
                str_constraints["max_length"] = prop["maxLength"]
            if "pattern" in prop:
                str_constraints["pattern"] = prop["pattern"]
            if str_constraints:
                from typing import Annotated

                from pydantic import StringConstraints

                py_type = Annotated[str, StringConstraints(**str_constraints)]

        # ── Array constraints: minItems / maxItems ──
        if json_type == "array":
            if "minItems" in prop:
                field_kwargs["min_length"] = prop["minItems"]
            if "maxItems" in prop:
                field_kwargs["max_length"] = prop["maxItems"]

        if name in required:
            fields[name] = (py_type, Field(**field_kwargs))
        else:
            fields[name] = (
                py_type | None,
                Field(default=None, **field_kwargs),
            )

    # Ensure at least one field — pydantic create_model requires fields
    if not fields:
        fields["result"] = (str, Field(default="", description="Output text"))

    return create_model(model_name, **fields, __base__=BaseModel)  # type: ignore[call-overload]