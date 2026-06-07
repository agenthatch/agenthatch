"""AgentHatch Capability Layer — v0.4 CapBus + type matching.

Defines the protocol contract between AHSSPEC.interface.provides/requires
and v0.4's CapBus runtime. v0.4 will register, match, and route capabilities.
"""

from agenthatch.cap.bus import CapabilityRegistration, CapBus
from agenthatch.cap.marshal import MatchResult, match_requires_to_provides

__all__ = [
    "CapBus",
    "CapabilityRegistration",
    "MatchResult",
    "match_requires_to_provides",
]
