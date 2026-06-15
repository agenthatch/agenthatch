"""SkillArchetype — deterministic skill classification.

Level 0 — classifies a skill into one of five archetypes based on
the AHSSpec interface (provides, requires, mcp_servers, api_templates,
scripts).  Drives BrickManifest generation in hatch Phase 2.5.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SkillArchetype(str, Enum):
    """Five mutually-exclusive skill archetypes."""
    PROMPT_ONLY = "prompt-only"         # no tools, no scripts
    TOOL_WRAPPER = "tool-wrapper"       # 1-2 simple tools
    MULTI_STEP = "multi-step"           # 3+ tools, scripts
    MCP_CONNECTOR = "mcp-connector"     # MCP server integration
    EXTERNAL_TOOL = "external-tool"     # wraps external CLI/binary


@dataclass
class ClassificationResult:
    """Result of classify_skill()."""
    archetype: SkillArchetype
    confidence: float
    reasons: list[str] = field(default_factory=list)


def classify_skill(spec: dict[str, Any] | Any) -> ClassificationResult:
    """Classify a skill into one of five archetypes.

    Decision order (first match wins):
    1. MCP servers in interface → mcp-connector
    2. External binary/tool requires with no script → external-tool
    3. API templates only, no scripts → tool-wrapper
    4. 3+ provides or scripts → multi-step
    5. 1-2 provides, no scripts → tool-wrapper
    6. No provides, no scripts → prompt-only

    Args:
        spec: AHSSpec dict, Pydantic model, or raw skill dict.

    Returns:
        ClassificationResult with archetype, confidence [0-1], and reasons.
    """
    interface = _get_interface(spec)
    resources = _get_resources(spec)

    provides = interface.get("provides", []) or []
    requires = interface.get("requires", []) or []
    mcp_servers = interface.get("mcp_servers", []) or []
    api_templates = interface.get("api_templates", []) or []
    scripts = resources.get("scripts", []) or []

    n_provides = len(provides)
    n_requires = len(requires)
    n_scripts = len(scripts)

    # Rule 1: MCP servers
    if mcp_servers:
        return ClassificationResult(
            archetype=SkillArchetype.MCP_CONNECTOR,
            confidence=0.95 if n_provides == 0 else 0.85,
            reasons=[f"{len(mcp_servers)} MCP server(s) detected"],
        )

    # Rule 2: External tool requires (CLI commands / binaries)
    external_requires = [
        r for r in requires
        if isinstance(r, dict) and r.get("type") in ("external", "binary", "cli")
    ]
    if external_requires and n_scripts == 0:
        return ClassificationResult(
            archetype=SkillArchetype.EXTERNAL_TOOL,
            confidence=0.90,
            reasons=[f"{len(external_requires)} external tool require(s)"],
        )

    # Rule 3: API templates only
    if api_templates and n_scripts == 0 and n_provides <= 2:
        return ClassificationResult(
            archetype=SkillArchetype.TOOL_WRAPPER,
            confidence=0.80 if n_provides > 0 else 0.70,
            reasons=[f"{len(api_templates)} API template(s)"],
        )

    # Rule 4: Multi-step (3+ provides or scripts)
    if n_provides >= 3 or n_scripts >= 3:
        return ClassificationResult(
            archetype=SkillArchetype.MULTI_STEP,
            confidence=0.85,
            reasons=[
                f"{n_provides} provides, {n_scripts} scripts"
            ],
        )

    # Rule 5: Tool wrapper (1-2 provides)
    if n_provides >= 1 or n_scripts >= 1:
        return ClassificationResult(
            archetype=SkillArchetype.TOOL_WRAPPER,
            confidence=0.75,
            reasons=[
                f"{n_provides} provides, {n_scripts} scripts"
            ],
        )

    # Rule 6: Prompt-only
    return ClassificationResult(
        archetype=SkillArchetype.PROMPT_ONLY,
        confidence=0.95,
        reasons=["No tools, scripts, or MCP servers"],
    )


def _get_interface(spec: dict[str, Any] | Any) -> dict[str, Any]:
    """Extract interface dict from any spec format."""
    if isinstance(spec, dict):
        return spec.get("interface", {}) or {}
    if hasattr(spec, "interface"):
        iface = spec.interface
        if hasattr(iface, "model_dump"):
            return iface.model_dump()
        if hasattr(iface, "dict"):
            return iface.dict()
        return iface
    return {}


def _get_resources(spec: dict[str, Any] | Any) -> dict[str, Any]:
    """Extract resources dict from any spec format."""
    if isinstance(spec, dict):
        return spec.get("resources", {}) or {}
    if hasattr(spec, "resources"):
        res = spec.resources
        if hasattr(res, "model_dump"):
            return res.model_dump()
        if hasattr(res, "dict"):
            return res.dict()
        return res
    return {}


def archetype_to_brick_config(
    archetype: SkillArchetype,
    api_templates: list[Any] | None = None,
    rules: list[Any] | None = None,
) -> dict[str, Any]:
    """Map a SkillArchetype to BrickManifest configuration flags.

    v0.8.22: Extracted from duplicated logic in generate/engine.py and
    agent/runtime.py.  Single source of truth for archetype → brick mapping.

    Returns dict with keys:
        loop_engine, capbus, hooks, credential_vault,
        file_processor, guard_active
    """
    from agenthatch_core.bricks.manifest import LoopKind  # deferred import

    api_templates = api_templates or []
    rules = rules or []

    return {
        "loop_engine": (
            LoopKind.DIRECT if archetype == SkillArchetype.PROMPT_ONLY
            else LoopKind.PLAN_GUIDED if archetype in (
                SkillArchetype.MULTI_STEP, SkillArchetype.MCP_CONNECTOR
            )
            else LoopKind.CONVERSATION
        ),
        "capbus": archetype != SkillArchetype.PROMPT_ONLY,
        "hooks": archetype not in (
            SkillArchetype.PROMPT_ONLY, SkillArchetype.EXTERNAL_TOOL
        ),
        "credential_vault": bool(api_templates),
        "file_processor": archetype in (
            SkillArchetype.TOOL_WRAPPER, SkillArchetype.MULTI_STEP
        ),
        "guard_active": bool(rules) and archetype != SkillArchetype.PROMPT_ONLY,
        # v0.9.8: task_complete_enabled — interactive REPL agents
        # (browser, shell, etc.) set this False so the user controls
        # when the session ends.  Default True for task-oriented agents.
        "task_complete_enabled": True,
        # v0.9.8: loop_workflow — step index to loop back to after
        # the linear workflow completes.  None means no loop.
        # Interactive agents typically set loop_steps=1.
        "loop_workflow": None,
    }
