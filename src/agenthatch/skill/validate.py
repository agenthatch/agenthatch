"""Schema validation + dual-layer targeted repair.

Layer 1: Pydantic AHSSpec.model_validate() — deterministic schema check.
Layer 2: Orchestrator targeted retry — only re-run the failing Harness.

Token savings (vs full re-run of all 5 Harnesses, ~8000 tokens):
  - identity.id format error → only Harness A (~500 tokens, save 94%)
  - interface.provides empty → only Harness C (~2000 tokens, save 75%)
  - base.runtime invalid → only Harness D (~1500 tokens, save 81%)
  - cross-field inconsistency → only Harness E (~1000 tokens, save 87%)
  - average: ~1250 tokens (~84% savings)

Draws from instructor's retry_sync loop pattern:
failure context + original inputs → LLM fix → validate → max 2 retries.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from agenthatch.skill.spec import (
    AHSSpec,
    BaseSpec,
    Composition,
    ContextPack,
    FileManifest,
    HarnessOutput,
    Identity,
    Instructions,
    Intent,
    Interface,
    _coerce_base_data,
)

logger = logging.getLogger("agenthatch")


# ─────────────────────────────────────────────────────────────────────────
# Targeted Repair Result
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class RepairResult:
    """Outcome of a targeted repair attempt."""
    ahs_spec: AHSSpec | None
    success: bool
    failure_fields: list[str] = field(default_factory=list)
    repair_attempts: int = 0
    total_tokens_saved: int = 0  # vs full re-run


# ─────────────────────────────────────────────────────────────────────────
# Field → Harness mapping (for targeted repair routing)
# ─────────────────────────────────────────────────────────────────────────

# Map AHS top-level fields to the Harness that produces them
_FIELD_TO_HARNESS: dict[str, str] = {
    "identity": "A",
    "intent": "B",
    "interface": "C",
    "base": "D",
    "instructions": "D",
}

# Map sub-fields to parent fields for Pydantic errors where loc[0] is a sub-field
_PARENT_FIELD_MAP: dict[str, str] = {
    # identity sub-fields
    "id": "identity", "display_name": "identity", "version": "identity",
    "license": "identity", "author": "identity", "meta": "identity",
    # intent sub-fields
    "triggers": "intent", "satisfies": "intent", "summary": "intent",
    # interface sub-fields
    "provides": "interface", "requires": "interface",
    "compatible_with": "interface", "mcp_servers": "interface",
    "api_templates": "interface",
    # instructions sub-fields
    "workflow": "instructions", "rules": "instructions",
    "safety": "instructions", "output_template": "instructions",
    "raw_body": "instructions",
    # base sub-fields
    "runtime": "base", "sandbox": "base", "timeout": "base",
    "env": "base", "dependencies": "base",
}

# Harness key → human-readable label
_HARNESS_LABEL: dict[str, str] = {
    "A": "extract_identity",
    "B": "infer_intent",
    "C": "infer_interface",
    "D": "detect_base_and_instructions",
    "E": "assemble_and_validate",
}


def validate_and_repair(
    ahs_dict: dict[str, Any],
    outputs: dict[str, HarnessOutput],
    harnesses: dict[str, Any],
    context: ContextPack,
    max_targeted_retries: int = 2,
) -> tuple[AHSSpec, dict[str, HarnessOutput]]:
    """Validate Pydantic schema and run targeted repair on failures.

    Dual-layer repair:
      1. Pydantic model_validate → identify failing fields
      2. Map failing fields → specific Harness → targeted re-run (not full re-run)
      3. Re-validate after each repair
      4. If still failing after max_targeted_retries → raise SchemaValidationError

    Args:
        ahs_dict: Raw dict from Harness E output (ahs_spec key).
        outputs: Current HarnessOutput dict from Orchestrator.
        harnesses: Dict of harness_name → AgentHarness instance.
        context: Original ContextPack (for Harness re-run inputs).
        max_targeted_retries: Max rounds of targeted repair (default 2).

    Returns:
        (validated AHSSpec, updated harness_outputs dict).

    Raises:
        SchemaValidationError: If repair exhausted without passing validation.
    """
    from agenthatch.exceptions import SchemaValidationError

    # Extract the ahs_spec dict
    spec_dict: dict[str, Any] = (
        ahs_dict.get("ahs_spec", ahs_dict) if isinstance(ahs_dict, dict) else ahs_dict
    )

    # ── v0.5.10: Coerce all AHSSPEC fields before validation ──
    if isinstance(spec_dict, dict) and spec_dict:
        from agenthatch.skill.spec import _coerce_ahs_dict
        spec_dict = _coerce_ahs_dict(spec_dict)

    total_saved = 0
    retries = 0

    while retries <= max_targeted_retries:
        # Step 1: Try Pydantic validation
        ahs_spec, errors = _try_validate(spec_dict)
        if ahs_spec is not None:
            # Attach confidence report and traces
            confidence_report = (
                ahs_dict.get("confidence_report", {})
                if isinstance(ahs_dict, dict)
                else {}
            )
            if confidence_report:
                from agenthatch.skill.spec import ConfidenceReport
                ahs_spec.confidence_report = ConfidenceReport(**confidence_report)
            ahs_spec.harness_traces = [
                outputs[k] for k in ["A", "B", "C", "D", "E"] if k in outputs
            ]
            logger.info(
                f"validate_and_repair: passed after {retries} repair rounds, "
                f"~{total_saved} tokens saved vs full re-run"
            )
            return ahs_spec, outputs

        # Step 2: Map errors to specific harnesses
        affected_harnesses = _map_errors_to_harnesses(errors)
        logger.warning(
            f"validate_and_repair round {retries + 1}: "
            f"{len(errors)} validation errors → harnesses {affected_harnesses}"
        )

        if not affected_harnesses:
            raise SchemaValidationError(
                f"Validation errors in non-Harness-mapped fields cannot be auto-repaired:\n"
                f"{_format_errors(errors)}\n"
                f"Manual AHSSPEC fix required for: "
                f"{[e.get('loc', 'unknown') for e in errors]}"
            )

        if retries >= max_targeted_retries:
            raise SchemaValidationError(
                f"AHSSPEC validation failed after {max_targeted_retries} targeted repair rounds.\n"
                f"Remaining errors: {_format_errors(errors)}"
            )

        # Step 3: Targeted re-run of affected harness(es)
        for harness_key in affected_harnesses:
            new_output, tokens = _retarget_harness(
                harness_key, harnesses, outputs, context
            )
            outputs[harness_key] = new_output
            total_saved += tokens

            # Update spec_dict with repaired field
            if harness_key == "A":
                spec_dict["identity"] = new_output.result.get("identity", {})
            elif harness_key == "B":
                spec_dict["intent"] = new_output.result.get("intent", {})
            elif harness_key == "C":
                spec_dict["interface"] = new_output.result.get("interface", {})
            elif harness_key == "D":
                spec_dict["base"] = new_output.result.get("base", {})
                spec_dict["instructions"] = new_output.result.get("instructions", {})

        # If E was in affected, re-run assembly
        if "E" in affected_harnesses:
            new_output, tokens = _retarget_harness(
                "E", harnesses, outputs, context
            )
            outputs["E"] = new_output
            total_saved += tokens
            spec_dict = new_output.result.get("ahs_spec", spec_dict)

        retries += 1

    # Should not reach here due to raise above, but for safety:
    raise SchemaValidationError(
        f"AHSSPEC validation failed after exhausting all repair attempts.\n"
        f"Errors: {_format_errors(errors)}"
    )


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _try_validate(spec_dict: dict[str, Any]) -> tuple[AHSSpec | None, list[dict[str, Any]]]:
    """Attempt Pydantic validation. Returns (AHSSpec, []) on success, (None, errors) on failure."""
    try:
        identity = Identity(**spec_dict.get("identity", {}))
        intent = Intent(**spec_dict.get("intent", {}))
        interface = Interface(**spec_dict.get("interface", {}))
        base = BaseSpec(**_coerce_base_data(spec_dict.get("base", {})))
        instructions = Instructions(**spec_dict.get("instructions", {}))
        composition = (
            Composition(**spec_dict.get("composition", {}))
            if spec_dict.get("composition")
            else Composition()
        )

        ahs_spec = AHSSpec(
            identity=identity,
            intent=intent,
            interface=interface,
            base=base,
            instructions=instructions,
            composition=composition,
        )
        # Also validate full model (cross-field)
        AHSSpec.model_validate(ahs_spec.model_dump())
        return ahs_spec, []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            errors.append({
                "loc": list(err["loc"]),
                "msg": err["msg"],
                "type": err["type"],
            })
        return None, errors
    except Exception as e:
        return None, [{"loc": ["__root__"], "msg": str(e), "type": "parse_error"}]


def _map_errors_to_harnesses(errors: list[dict[str, Any]]) -> list[str]:
    """Map validation error locations to affected harness keys."""
    harnesses: set[str] = set()
    for err in errors:
        loc = err.get("loc", [])
        if not loc:
            continue
        top_field = str(loc[0])
        harness_key = _FIELD_TO_HARNESS.get(top_field)
        if not harness_key:
            # Try sub-field → parent field resolution
            parent = _PARENT_FIELD_MAP.get(top_field)
            if parent:
                harness_key = _FIELD_TO_HARNESS.get(parent)
        if not harness_key and top_field == "__root__":
            harness_key = "E"  # parse error → re-run assembly
        if harness_key:
            harnesses.add(harness_key)
    return sorted(harnesses)


def _retarget_harness(
    harness_key: str,
    harnesses: dict[str, Any],
    outputs: dict[str, HarnessOutput],
    context: ContextPack,
) -> tuple[HarnessOutput, int]:
    """Re-run a specific harness with its original inputs.

    Returns:
        (new HarnessOutput, estimated_token_savings).
    """
    h = harnesses.get(harness_key)
    if h is None:
        return (
            HarnessOutput(
                result={},
                confidence=0.0,
                reasoning_trace=[f"Harness {harness_key} not found"],
                self_check_passed=False,
                degradation_applied=["harness_not_found"],
            ),
            0,
        )

    logger.info(f"Retargeting Harness {harness_key} ({_HARNESS_LABEL.get(harness_key, '?')})")

    if harness_key == "A":
        output = h.run(
            frontmatter=context.frontmatter,
            dir_name=context.dir_name,
            body_first_50_lines=context.body[:2500],
            file_contents=context.file_manifest.content_bundle(),
        )
    elif harness_key == "B":
        output = h.run(
            description=context.frontmatter.get("description") if context.frontmatter else None,
            body=context.body,
            frontmatter_name=context.frontmatter.get("name") if context.frontmatter else None,
            file_contents=context.file_manifest.content_bundle(),
        )
    elif harness_key == "C":
        file_contents = context.file_manifest.content_bundle()
        output = h.run(
            body=context.body,
            file_contents=file_contents,
            frontmatter_allowed_tools=(
                context.frontmatter.get("allowed_tools") if context.frontmatter else None
            ),
        )
    elif harness_key == "D":
        file_contents = context.file_manifest.content_bundle()
        output = h.run(
            body=context.body,
            file_contents=file_contents,
            frontmatter_compatibility=(
                context.frontmatter.get("compatibility") if context.frontmatter else None
            ),
            frontmatter_allowed_tools=(
                context.frontmatter.get("allowed_tools") if context.frontmatter else None
            ),
        )
    elif harness_key == "E":
        resources = _build_validate_resources(context.file_manifest)
        output = h.run(
            identity=outputs["A"].result if "A" in outputs else {},
            intent=outputs["B"].result if "B" in outputs else {},
            interface=outputs["C"].result if "C" in outputs else {},
            base=outputs["D"].result.get("base", {}) if "D" in outputs else {},
            instructions=outputs["D"].result.get("instructions", {}) if "D" in outputs else {},
            resources=resources,
            dir_name=context.dir_name,
        )
    else:
        output = h.run()

    savings = _estimate_token_savings(harness_key)
    return output, savings


def _estimate_token_savings(harness_key: str) -> int:
    """Estimate token savings vs full 5-Harness re-run (~8000 tokens)."""
    estimates = {
        "A": 7500,   # 94% savings → ~500 tokens
        "B": 6000,   # 75% savings → ~2000 tokens
        "C": 6000,   # 75% savings → ~2000 tokens
        "D": 6500,   # 81% savings → ~1500 tokens
        "E": 7000,   # 87% savings → ~1000 tokens
    }
    return estimates.get(harness_key, 0)


def _format_errors(errors: list[dict[str, Any]]) -> str:
    """Format validation errors for display."""
    lines: list[str] = []
    for err in errors:
        loc = " → ".join(str(x) for x in err.get("loc", []))
        msg = err.get("msg", "unknown error")
        lines.append(f"  - {loc}: {msg}")
    return "\n".join(lines)


def _build_validate_resources(manifest: FileManifest) -> dict[str, list[dict[str, str]]]:
    """Build resources dict from flat file manifest for validation repair.

    Same logic as engine.py's _build_resources — groups by extension.
    """
    from pathlib import Path

    _SCRIPT_SUFFIXES = {".py", ".sh", ".js", ".ts", ".rb", ".go", ".rs"}
    _REF_SUFFIXES = {".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml"}

    scripts: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    assets: list[dict[str, str]] = []

    for e in manifest.entries:
        if e.path == manifest.entrypoint:
            continue
        suffix = Path(e.path).suffix.lower()
        entry_dict = {"name": e.path, "hash": e.hash}
        if suffix in _SCRIPT_SUFFIXES:
            scripts.append(entry_dict)
        elif suffix in _REF_SUFFIXES:
            references.append(entry_dict)
        else:
            assets.append(entry_dict)

    return {"scripts": scripts, "references": references, "assets": assets}
