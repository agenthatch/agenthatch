"""Builtin capability primitives (v0.4).

Each builtin is an independent Python class implementing the BuiltinCapability
abstract base. The BUILTIN_REGISTRY maps capability names to their classes.
"""

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any


def with_enriched_errors(method: Callable[..., str]) -> Callable[..., str]:
    """Decorator: enrich TypeError with accepted parameter names."""
    sig = inspect.signature(method)
    accepted_params = list(sig.parameters.keys())

    @functools.wraps(method)
    def wrapper(*args: Any, **kwargs: Any) -> str:
        try:
            return method(*args, **kwargs)
        except TypeError as e:
            msg = str(e)
            if "unexpected keyword argument" in msg:
                rejected = msg.split("'")[1] if "'" in msg else "unknown"
                return (
                    f"Error: '{rejected}' is not accepted. "
                    f"Accepted parameters: {accepted_params}. "
                    f"Try one of: {', '.join(accepted_params)}"
                )
            return f"Error: {msg}. Accepted parameters: {accepted_params}"
    return wrapper


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

# ── Eagerly load all builtin modules to populate BUILTIN_REGISTRY ──
from agenthatch.agent.builtins import (  # noqa: E402, F401
    file_io,
    geolocation,
    http_client,
    json_parser,
    runtime,
    template_renderer,
    text_synthesis,
    web_search,
)
