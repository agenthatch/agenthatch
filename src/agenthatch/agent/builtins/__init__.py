"""Builtin capability primitives (v0.4).

Each builtin is an independent Python class implementing the BuiltinCapability
abstract base. The BUILTIN_REGISTRY maps capability names to their classes.
"""

from abc import ABC, abstractmethod


class BuiltinCapability(ABC):
    """Base class for builtin capabilities."""
    name: str = ""
    cap_type: str = ""
    description: str = ""
    schema: dict = {}

    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the capability."""
        ...


BUILTIN_REGISTRY: dict[str, type[BuiltinCapability]] = {}
