"""Capability type matching engine — v0.4 runtime matching."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class MatchResult:
    """Result of a capability match."""
    matched: bool
    provider_skill: str | None = None
    capability: str = ""
    match_type: str = ""


def match_requires_to_provides(
    requires: list[dict[str, Any]],
    available: dict[str, list[str]],
) -> list[MatchResult]:
    """Match requires to available provides."""
    results: list[MatchResult] = []
    for req in requires:
        cap_name = req.get("capability", "")
        if cap_name in available:
            results.append(MatchResult(
                matched=True,
                provider_skill=available[cap_name][0],
                capability=cap_name,
                match_type="exact",
            ))
        else:
            results.append(MatchResult(
                matched=False,
                capability=cap_name,
                match_type="none",
            ))
    return results


def fuzzy_match(required: str, capabilities: dict[str, Any]) -> Any | None:
    """Fuzzy match a capability name to registered capabilities.

    Normalizes underscores↔hyphens and does case-insensitive comparison.
    Returns the best-match CapabilityRegistration or None.
    """
    normalized = required.lower().replace("_", "-")
    for name, cap in capabilities.items():
        cap_normalized = name.lower().replace("_", "-")
        if normalized == cap_normalized:
            return cap
    return None
