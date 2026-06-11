"""Phase 2 Harness Reflection — v0.8 structured meta-agent introspection.

Each harness produces an output, then enters a structured reflection round
with three mandatory dimensions: self-consistency, cross-harness consistency,
and fidelity to source.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Reflection prompts
# ─────────────────────────────────────────────────────────────────────────

HARNESS_REFLECTION_PROMPT = """\
You have just produced an output for a skill inference task. Now you must REFLECT.

## Your Output
{harness_output_json}

## Original Source (SKILL.md)
{skill_md_body}

## Peer Harness Outputs (for cross-validation)
{peer_outputs}

## REFLECTION CHECKLIST
Answer each with PASS or FAIL and provide specific reasoning:

### 1. SELF-CONSISTENCY
Does my output internally contradict itself? Are all values within their allowed enums/types?
Are all required fields present? Are any fields mutually exclusive but both set?

### 2. CROSS-HARNESS CONSISTENCY
Do my inferred values align with what peer harnesses produced? Flag any direct conflicts.
- If I infer capability type X, does the intent harness describe an intent matching X?
- If I detect base rule Y, does the assembly harness include Y in its workflow?

### 3. FIDELITY TO SOURCE
- Is anything in my output NOT evidenced in the SKILL.md source? (FLAG AS HALLUCINATION)
- Is anything in the SKILL.md source MISSING from my output? (FLAG AS OMISSION)

## Correction Instructions
If any check FAILS, provide the corrected value in the `corrections` field.
Each correction must include {{field, old_value, new_value, reason}}.
"""

HARNESS_REFLECTION_FEW_SHOT = """\
## Example: Self-consistency failure
Identity output had `version: "v1.0.0"` but SKILL.md says `version: "1.0"`.
→ Self-consistency: FAIL
→ Correction: {{field: "identity.version", old: "v1.0.0", new: "1.0",
  reason: "SKILL.md does not use v-prefix"}}

## Example: Cross-harness conflict
Interface harness says capability "search_files" has type="action".
Intent harness says this skill is for "knowledge retrieval", not "action execution".
→ Cross-harness: FAIL
→ Correction: {{field: "provides[search_files].type", old: "action",
  new: "knowledge", reason: "Intent harness classifies skill as
  knowledge retrieval"}}

## Example: Fidelity omission
SKILL.md describes a "download_document" capability but interface output omits it.
→ Fidelity: FAIL — omission
→ Correction: {{add capability "download_document" with type="data"}}
"""


# ─────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────

class CorrectionItem(BaseModel):
    """A single correction from a reflection round."""
    field: str          # Dot-notation path, e.g. "provides[0].type"
    old_value: str | None = None
    new_value: str
    reason: str


class HarnessReflectionOutput(BaseModel):
    """Structured reflection output from a single harness."""
    self_consistency: Literal["pass", "fail"]
    self_consistency_detail: str = ""
    cross_harness: Literal["pass", "fail"]
    cross_harness_detail: str = ""
    fidelity: Literal["pass", "fail"]
    fidelity_detail: str = ""
    corrections: list[CorrectionItem] = Field(default_factory=list)
    overall_confidence: float = Field(
        ge=0.0, le=1.0,
        description="Updated confidence after reflection",
    )


# ─────────────────────────────────────────────────────────────────────────
# Fidelity Checkpoint CP2
# ─────────────────────────────────────────────────────────────────────────

FIDELITY_CHECKPOINT_PROMPT = """\
You are verifying that an AHSSPEC faithfully represents the original SKILL.md.

## Original SKILL.md
{skill_md}

## Generated AHSSPEC
{ahspec_json}

## CHECKLIST
1. COVERAGE: List every capability, intent, and rule from SKILL.md.
   Does AHSSPEC cover ALL of them? (PASS/FAIL + detail)
2. HALLUCINATION: List every capability, intent, and rule in AHSSPEC.
   Is each one evidenced in SKILL.md? (PASS/FAIL + detail)
3. FIDELITY_SCORE: 0.0-1.0
   - 1.0 = perfect coverage, no hallucinations
   - 0.7 = minor omissions or over-specificity
   - < 0.5 = major gaps or fabrications
"""


class FidelityCheckpointOutput(BaseModel):
    """Output from the Phase 2→3 fidelity checkpoint."""
    coverage: Literal["pass", "fail"]
    coverage_detail: str = ""
    hallucination: Literal["pass", "fail"]
    hallucination_detail: str = ""
    fidelity_score: float = Field(ge=0.0, le=1.0)
    recommendations: list[str] = Field(default_factory=list)


class FidelityCriticalError(Exception):
    """Raised when CP2 fidelity score is critically low."""
    pass


# ─────────────────────────────────────────────────────────────────────────
# Dataclass helpers
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ReflectionContext:
    """Context needed for a harness reflection round."""
    harness_output_json: str      # Current harness's output as JSON
    skill_md_body: str             # SKILL.md body (first 3000 chars)
    peer_outputs: str              # Other harness outputs as JSON
    harness_name: str              # "A", "B", "C", "D", "E", or "F"


@dataclass
class ReflectionResult:
    """The result of both primary inference and reflection."""
    output: dict[str, Any]         # Final (possibly corrected) output
    raw_output: dict[str, Any]     # Original pre-reflection output
    reflection: HarnessReflectionOutput | None = None
    confidence: float = 0.0
    corrections_applied: int = 0


# ─────────────────────────────────────────────────────────────────────────
# Reflection runner
# ─────────────────────────────────────────────────────────────────────────

def run_harness_reflection(
    client: Any,                    # LLMClient
    model: str,
    ctx: ReflectionContext,
) -> HarnessReflectionOutput:
    """Run structured reflection for a single harness output.

    Args:
        client: LLMClient instance
        model: Model name
        ctx: ReflectionContext with output, source, and peer data

    Returns:
        HarnessReflectionOutput with pass/fail and corrections
    """
    prompt = HARNESS_REFLECTION_PROMPT.format(
        harness_output_json=ctx.harness_output_json[:3000],
        skill_md_body=ctx.skill_md_body[:3000],
        peer_outputs=ctx.peer_outputs[:2000],
    )

    try:
        reflection = client.chat_structured(
            messages=[
                {"role": "system", "content": HARNESS_REFLECTION_FEW_SHOT},
                {"role": "user", "content": prompt},
            ],
            response_model=HarnessReflectionOutput,
            model=model,
            temperature=0.2,
            thinking=True,
        )
        return reflection  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Harness reflection failed for %s: %s", ctx.harness_name, e)
        return HarnessReflectionOutput(
            self_consistency="pass",
            self_consistency_detail="reflection LLM call failed — accepting output as-is",
            cross_harness="pass",
            cross_harness_detail="",
            fidelity="pass",
            fidelity_detail="",
            corrections=[],
            overall_confidence=0.5,
        )


def apply_corrections(
    output: dict[str, Any],
    corrections: list[CorrectionItem],
) -> dict[str, Any]:
    """Apply correction items to a harness output dict.

    Supports dot-notation field paths like "identity.version" or
    "provides[0].type" by splitting on dots and navigating nested dicts.
    """
    import copy

    result = copy.deepcopy(output)
    for corr in corrections:
        parts = corr.field.split(".")
        container: dict[str, Any] = result
        for _i, part in enumerate(parts[:-1]):
            # Handle array index like "provides[0]"
            if "[" in part and part.endswith("]"):
                key, idx_str = part.split("[", 1)
                idx_str = idx_str.rstrip("]")
                try:
                    idx = int(idx_str)
                    container = container[key][idx]
                except (KeyError, IndexError, ValueError, TypeError):
                    break
            else:
                if part not in container:
                    container = {}
                    break
                container = container[part]

        if isinstance(container, dict) and parts:
            last_part = parts[-1]
            # Handle array index in last part
            if "[" in last_part and last_part.endswith("]"):
                key, idx_str = last_part.split("[", 1)
                idx_str = idx_str.rstrip("]")
                try:
                    idx = int(idx_str)
                    container[key][idx] = corr.new_value
                except (KeyError, IndexError, ValueError, TypeError):
                    pass
            else:
                try:
                    container[last_part] = corr.new_value
                except TypeError:
                    pass

    return result


def should_skip_reflection(
    harness_name: str,
    confidence: float,
    has_errors: bool = False,
) -> bool:
    """Determine if reflection can be skipped for a harness.

    Optimization: Harnesses A and F can skip reflection if primary
    confidence > 0.9. Harness E can skip if all cross-checks pass.
    """
    if has_errors:
        return False  # Never skip if there were errors

    if harness_name in ("A", "F") and confidence >= 0.9:
        return True
    if harness_name == "E" and confidence >= 0.95:
        return True

    return False
