"""Capability type matching engine — stub for v0.4 runtime.

Matches AHSSPEC.interface.requires[i] against registered provides
based on capability name + type compatibility.

v0.4 will provide complete type matching and compatibility resolution.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchResult:
    """Result of a capability match."""
    matched: bool
    provider_skill: str | None = None
    capability: str = ""
    match_type: str = ""  # "exact", "compatible", "none"


def match_requires_to_provides(
    requires: list[dict[str, Any]],
    available: dict[str, list[str]],  # capability → [skill_id, ...]
) -> list[MatchResult]:
    """Match requires to available provides. (v0.4 implementation)

    Args:
        requires: List of requirement dicts with capability + type.
        available: Map of capability name → list of providing skill IDs.

    Returns:
        List of MatchResult, one per requirement.
    """
    return []
