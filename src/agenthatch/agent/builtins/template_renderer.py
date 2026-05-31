"""Template renderer builtin capability."""

from typing import Any

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class TemplateRendererCap(BuiltinCapability):
    name = "template_renderer"
    cap_type = "formatter"
    description = "Render text templates with variables"
    schema = {
        "type": "object",
        "properties": {
            "template": {"type": "string", "description": "Template string with {placeholders}"},
            "variables": {"type": "object", "description": "Variable name → value mapping"},
        },
        "required": ["template", "variables"],
    }

    def execute(self, template: str = "", variables: dict[str, Any] | None = None) -> str:  # type: ignore[override]
        if variables is None:
            variables = {}
        try:
            return template.format(**variables)
        except KeyError as e:
            return f"Error: missing variable {e}"
        except Exception as e:
            return f"Error rendering template: {e}"


BUILTIN_REGISTRY["template_renderer"] = TemplateRendererCap
