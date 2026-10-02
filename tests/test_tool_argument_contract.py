"""v1.0.25: the tool argument contract must tell the LLM the truth.

Three layers are locked here:
1. Schema truth — dict params declare "object", list params "array"
   (_annotation_to_json_type / _signature_to_parameters_schema).
2. Argument coercion — JSON-stringified object/array/number arguments
   are revived before the call; anything unparsable passes through
   unchanged; no eval.
3. Actionable failure — on TypeError/ValueError the LLM gets a retry
   hint with the expected types instead of a bare Python traceback line
   it cannot act on.
"""
import inspect
from typing import Any, Optional

from agenthatch_core.agent import (
    _annotation_to_json_type,
    _coerce_arguments,
    _make_python_tool_executor,
    _signature_to_parameters_schema,
)
from agenthatch_core.tools.bus import CapBus


class TestAnnotationToJsonType:
    def test_type_matrix(self):
        assert _annotation_to_json_type(dict) == "object"
        assert _annotation_to_json_type(dict[str, Any]) == "object"
        assert _annotation_to_json_type(list) == "array"
        assert _annotation_to_json_type(list[str]) == "array"
        assert _annotation_to_json_type(str) == "string"
        assert _annotation_to_json_type(int) == "integer"
        assert _annotation_to_json_type(float) == "number"
        assert _annotation_to_json_type(bool) == "boolean"
        assert _annotation_to_json_type(Optional[dict]) == "object"
        assert _annotation_to_json_type(inspect.Parameter.empty) == "string"
        assert _annotation_to_json_type(Any) == "string"
        # string annotations (from __future__ import annotations modules)
        assert _annotation_to_json_type("dict") == "object"
        assert _annotation_to_json_type("list[str]") == "array"

    def test_schema_declares_object_and_array(self):
        """The live v1.0.24 failure: a dict param was declared 'string'."""

        def tool(parameters: dict = None, palette: list = None, name: str = None):
            ...

        schema = _signature_to_parameters_schema(inspect.signature(tool))
        assert schema["properties"]["parameters"]["type"] == "object"
        assert schema["properties"]["palette"]["type"] == "array"
        assert schema["properties"]["name"]["type"] == "string"
        assert schema["required"] == []


class TestCoerceArguments:
    def test_json_string_revived_to_dict(self):
        def tool(parameters: dict = None): ...

        sig = inspect.signature(tool)
        out = _coerce_arguments(sig, {"parameters": '{"seed": 123}'})
        assert out["parameters"] == {"seed": 123}

    def test_json_string_revived_to_list(self):
        def tool(palette: list = None): ...

        sig = inspect.signature(tool)
        out = _coerce_arguments(sig, {"palette": '["#fff", "#000"]'})
        assert out["palette"] == ["#fff", "#000"]

    def test_numeric_string_coerced(self):
        def tool(seed: float = None): ...

        sig = inspect.signature(tool)
        assert _coerce_arguments(sig, {"seed": "7.5"})["seed"] == 7.5

    def test_unparsable_string_passes_through(self):
        def tool(parameters: dict = None): ...

        sig = inspect.signature(tool)
        out = _coerce_arguments(sig, {"parameters": "not json"})
        assert out["parameters"] == "not json"

    def test_wrong_shape_parsed_passes_through(self):
        """json.loads succeeds but yields a non-dict — must NOT inject it."""

        def tool(parameters: dict = None): ...

        sig = inspect.signature(tool)
        out = _coerce_arguments(sig, {"parameters": "123"})
        assert out["parameters"] == "123"

    def test_non_string_values_untouched(self):
        def tool(parameters: dict = None): ...

        sig = inspect.signature(tool)
        out = _coerce_arguments(sig, {"parameters": {"a": 1}})
        assert out["parameters"] == {"a": 1}


class TestActionableFailure:
    def test_retry_hint_on_value_error(self):
        """The exact v1.0.24 E2E crash: dict(str) on an unparsable input."""

        def fragile_tool(parameters: dict = None, seed: float = None):
            return dict(parameters)

        executor = _make_python_tool_executor(fragile_tool)
        result = executor({"parameters": "garbage"})
        assert result.startswith("Error: tool 'fragile_tool' failed")
        assert "'parameters'=object" in result
        assert "Retry with JSON-native values" in result

    def test_coercion_prevents_the_live_crash(self):
        """A JSON-stringified dict argument must reach the tool as a dict."""

        def art_tool(parameters: dict = None):
            return dict(parameters) if parameters else {}

        executor = _make_python_tool_executor(art_tool)
        assert executor({"parameters": '{"particleCount": 1800}'}) == {"particleCount": 1800}

    def test_success_passes_through_untouched(self):
        def simple_tool(name: str = None):
            return f"hello {name}"

        executor = _make_python_tool_executor(simple_tool)
        assert executor({"name": "world"}) == "hello world"

    def test_bus_end_to_end_with_coercion(self):
        def art_tool(parameters: dict = None):
            return {"ok": True, "params": parameters}

        bus = CapBus(task_complete_enabled=False)
        bus.register(
            name="art_tool",
            executor=_make_python_tool_executor(art_tool),
            schema={
                "name": "art_tool",
                "description": "d",
                "parameters": _signature_to_parameters_schema(
                    inspect.signature(art_tool)
                ),
            },
            source="user",
        )
        out = bus.route("art_tool", {"parameters": '{"particleCount": 1800}'})
        assert isinstance(out, dict)
        assert out["params"] == {"particleCount": 1800}
