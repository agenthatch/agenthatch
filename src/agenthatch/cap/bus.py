"""Capability Bus — interface definition for v0.4 runtime brick communication.

v0.3 defines the protocol contract (DD-008) between AHSSPEC.interface.provides/requires
and v0.4's CapBus runtime. v0.4 will register, match, and route capabilities.

Protocol:
  v0.3 AHSSPEC.interface.provides[i].capability
    ≡ v0.4 CapBus.register(name=...) 第一个参数

  v0.3 AHSSPEC.interface.requires[i].capability
    ≡ v0.4 Resolver.match_builtin(name) 或 skillhouse.find_provider(name) 的查询 key
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapabilityRegistration:
    """A capability registered on the bus."""
    name: str
    type: str  # data, analysis, media, transform, action, event, knowledge, renderer
    schema: dict[str, Any] = field(default_factory=dict)
    source_skill: str = ""


@dataclass
class CapBus:
    """Capability Bus — stub for v0.4 runtime implementation.

    v0.4 will provide complete register/match/route/invoke implementations.
    v0.3 only defines the interface contract.
    """

    capabilities: dict[str, CapabilityRegistration] = field(default_factory=dict)
    builtins: set[str] = field(default_factory=set)

    def register(
        self,
        name: str,
        cap_type: str,
        schema: dict[str, Any] | None = None,
        source_skill: str = "",
    ) -> None:
        """Register a capability on the bus. (v0.4 implementation)"""
        pass

    def match(self, required: str) -> CapabilityRegistration | None:
        """Match a required capability to a registered provider. (v0.4 implementation)"""
        return None

    def inject_builtin(self, name: str) -> None:
        """Inject a builtin capability. (v0.4 implementation)"""
        pass
