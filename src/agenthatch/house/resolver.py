"""Topology resolver — dependency resolution for v0.4 brick assembly.

Provides topological sort and provides↔requires matching.
Consumed by v0.4 SkillAgent.from_ahspec() for dependency resolution.

v0.3 provides the topological_sort algorithm. v0.4 extends with
runtime matching and builtin capability injection.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from agenthatch.exceptions import DependencyCycleError


def resolve_dependencies(
    required_capabilities: list[str],
    providers: dict[str, list[str]],
    dependency_graph: dict[str, list[str]],
    entries: dict[str, Any],
) -> list[str]:
    """Resolve dependency order for required capabilities.

    Args:
        required_capabilities: Capability names needed.
        providers: Map of capability → [skill_id, ...].
        dependency_graph: Map of skill_id → [capability_name, ...].
        entries: All indexed skill entries.

    Returns:
        Ordered list of skill IDs to assemble.
    """
    if not required_capabilities:
        return []

    # Find which skills provide the required capabilities
    needed_skills: set[str] = set()
    for cap in required_capabilities:
        for skill_id in providers.get(cap, []):
            needed_skills.add(skill_id)

    if not needed_skills:
        return []

    # Build skill-to-skill adjacency by resolving capability names → skill IDs
    adjacency: dict[str, set[str]] = {sid: set() for sid in needed_skills}
    rev_adjacency: dict[str, set[str]] = {sid: set() for sid in needed_skills}
    for sid in needed_skills:
        for cap in dependency_graph.get(sid, []):
            for provider_sid in providers.get(cap, []):
                if provider_sid in needed_skills:
                    adjacency[sid].add(provider_sid)
                    rev_adjacency.setdefault(provider_sid, set()).add(sid)

    # Build in-degree map
    in_degree: dict[str, int] = {
        sid: len(adjacency[sid]) for sid in needed_skills
    }

    # Kahn's algorithm
    queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
    result: list[str] = []

    while queue:
        node = queue.popleft()
        result.append(node)
        for dependent in rev_adjacency.get(node, ()):
            in_degree[dependent] -= 1
            if in_degree[dependent] == 0:
                queue.append(dependent)

    if len(result) != len(needed_skills):
        missing = set(needed_skills) - set(result)
        raise DependencyCycleError(
            f"Circular dependency detected among skills: {missing}"
        )
    return result


def is_builtin(capability: str) -> bool:
    """Check if a capability is a builtin (from infrastructure_catalog).

    v0.4 will maintain the full catalog; v0.3 provides a minimal set.
    """
    from agenthatch.skill.prompts import FLAT_CATALOG
    return capability in FLAT_CATALOG


def wire(
    capability: str,
    source_skill: str,
    dependency_graph: dict[str, list[str]],
) -> None:
    """Wire a capability dependency between skills.

    v0.4: Records that `capability` is provided by `source_skill` in the
    dependency graph, enabling runtime cross-skill tool routing.
    """
    if source_skill not in dependency_graph:
        dependency_graph[source_skill] = []
    if capability not in dependency_graph[source_skill]:
        dependency_graph[source_skill].append(capability)
