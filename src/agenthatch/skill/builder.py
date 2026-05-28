"""Full pipeline orchestration — Phase 1 + Phase 2 + Post-Process.

Ties together the complete agenthatch v0.3 flow:
  1. Phase 1: ContextPack from parser
  2. Phase 2: AHSSpec from Orchestrator + 5 AgentHarnesses
  3. Post-Process: Validation + skillhouse registration
"""

from __future__ import annotations

import logging
from typing import Any

from agenthatch.skill.engine import Orchestrator
from agenthatch.skill.parser import assemble_context as parser_assemble_context
from agenthatch.skill.spec import AHSSpec, ContextPack, HarnessOutput

logger = logging.getLogger("agenthatch")


def build_ahspec_from_path(
    skill_path: str,
    config: dict[str, Any] | None = None,
    large_model: str = "",
    small_model: str = "",
) -> tuple[AHSSpec, dict[str, HarnessOutput]]:
    """Full pipeline: skill path → AHSSpec.

    Convenience wrapper that runs both phases from a file path.

    Args:
        skill_path: Path to skill directory or SKILL.md file.
        config: Optional config dict (loaded from Config.load() if None).
        large_model: Override for large model tier.
        small_model: Override for small model tier.

    Returns:
        (validated AHSSpec, harness_outputs dict).
    """
    from agenthatch.config import Config

    if config is None:
        config = Config.load()

    # Phase 1
    context = parser_assemble_context(skill_path)
    logger.info(
        f"build_ahspec: Phase 1 complete — dir_name={context.dir_name}, "
        f"files={len(context.file_manifest.entries)}"
    )

    # Phase 2
    return build_ahspec(context, config, large_model=large_model, small_model=small_model)


def build_ahspec(
    context: ContextPack,
    config: dict[str, Any],
    large_model: str = "",
    small_model: str = "",
) -> tuple[AHSSpec, dict[str, HarnessOutput]]:
    """Run Phase 2 on a ContextPack.

    Args:
        context: ContextPack from Phase 1.
        config: Full config dict.
        large_model: Override for large model tier.
        small_model: Override for small model tier.

    Returns:
        (validated AHSSpec, harness_outputs dict).
    """
    orchestrator = Orchestrator(
        config,
        large_model=large_model,
        small_model=small_model,
    )
    return orchestrator.run(context)
