"""Builtin capability primitives (v0.4).

Each builtin is an independent Python class implementing the BuiltinCapability
abstract base. The BUILTIN_REGISTRY maps capability names to their classes.
"""

from abc import ABC, abstractmethod
from typing import Any


class BuiltinCapability(ABC):
    """Base class for builtin capabilities."""
    name: str = ""
    cap_type: str = ""
    description: str = ""
    schema: dict[str, Any] = {}

    @abstractmethod
    def execute(self, **kwargs: Any) -> str:
        """Execute the capability."""
        ...


BUILTIN_REGISTRY: dict[str, type[BuiltinCapability]] = {}
