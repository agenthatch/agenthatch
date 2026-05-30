"""Geolocation builtin capability."""

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class GeolocationCap(BuiltinCapability):
    name = "geolocation"
    cap_type = "service"
    description = "Resolve location names to coordinates and vice versa"
    schema = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Location name or coordinates"},
        },
        "required": ["location"],
    }

    def execute(self, location: str = "") -> str:
        return (
            f"Geolocation resolution for '{location}' is not yet implemented. "
            "This capability requires a geocoding API key."
        )


BUILTIN_REGISTRY["geolocation"] = GeolocationCap
