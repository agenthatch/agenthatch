"""Phase 2 Harness Reflection — v0.8 structured meta-agent introspection.

Each harness produces an output, then enters a structured reflection round
with three mandatory dimensions: self-consistency, cross-harness consistency,
and fidelity to source.

v0.9.20: Activated — ``reflect_and_correct_harness`` and
``run_fidelity_checkpoint`` are now wired into ``Orchestrator.run()``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agenthatch.skill.spec import HarnessOutput

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

    v0.9.20 safety guards:
    - **Type auto-conversion**: if ``new_value`` is a JSON string but the
      target field is currently a list/dict, ``json.loads`` is attempted.
      This prevents LLMs returning ``'["a","b"]'`` (string) from replacing
      a native list with a string.
    - **old_value match check**: if ``old_value`` is non-empty, the
      current field value must match it. If not, the correction is based
      on a wrong premise and is silently skipped.
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
                    current = container[key][idx]
                except (KeyError, IndexError, ValueError, TypeError):
                    continue
                # old_value match check (coerce to current type for list/dict)
                if corr.old_value:
                    expected_old = _coerce_value(corr.old_value, current)
                    if current != expected_old:
                        continue
                # Type auto-conversion
                new_val = _coerce_value(corr.new_value, current)
                container[key][idx] = new_val
            else:
                if last_part not in container:
                    continue
                current = container[last_part]
                # old_value match check (coerce to current type for list/dict)
                if corr.old_value:
                    expected_old = _coerce_value(corr.old_value, current)
                    if current != expected_old:
                        continue
                # Type auto-conversion
                new_val = _coerce_value(corr.new_value, current)
                container[last_part] = new_val

    return result


def _coerce_value(new_value: Any, current: Any) -> Any:
    """Coerce ``new_value`` to match the type of ``current`` when safe.

    If ``new_value`` is a string but ``current`` is a list or dict,
    attempt ``json.loads`` to parse it into the native type.
    Returns ``new_value`` unchanged if coercion is not applicable or fails.
    """
    if isinstance(new_value, str) and isinstance(current, (list, dict)):
        stripped = new_value.strip()
        if stripped and stripped[0] in "[{":
            try:
                return json.loads(stripped)
            except (json.JSONDecodeError, TypeError):
                pass
    return new_value


def _detect_type_change(original: Any, corrected: Any, path: str = "") -> str:
    """Detect if a correction changed a field's Python type.

    Recursively compares ``original`` and ``corrected``. Returns a
    human-readable path string (e.g. ``"intent.triggers"``) for the
    first type mismatch found, or empty string if all types are
    preserved.

    Type changes caught:
    - ``list`` → ``str`` (e.g. JSON string not coerced)
    - ``dict`` → ``str``
    - ``list`` → ``dict`` or vice versa
    - ``int``/``float`` → ``str``
    Note: ``str`` → ``str`` (value change) is NOT flagged here — that
    is for ``validate_output`` to catch.
    """
    if type(original) is not type(corrected):
        # Allow int ↔ float (both numeric)
        if isinstance(original, (int, float)) and isinstance(corrected, (int, float)):
            pass
        else:
            return path or "(root)"

    if isinstance(original, dict) and isinstance(corrected, dict):
        for key in original:
            if key in corrected:
                child_path = f"{path}.{key}" if path else key
                result = _detect_type_change(original[key], corrected[key], child_path)
                if result:
                    return result

    if isinstance(original, list) and isinstance(corrected, list):
        if original and corrected:
            child_path = f"{path}[0]" if path else "[0]"
            result = _detect_type_change(original[0], corrected[0], child_path)
            if result:
                return result

    return ""


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


# ─────────────────────────────────────────────────────────────────────────
# v0.9.20: Reflection-correction loop orchestrators
# ─────────────────────────────────────────────────────────────────────────

def _accumulate_usage(
    accumulated: dict[str, int], client: Any
) -> dict[str, int]:
    """Read ``client.last_usage`` and add to the running token total.

    Mirrors ``engine._accumulate_token_usage`` but kept local to avoid a
    circular import (engine imports reflection, not vice-versa).
    """
    usage = getattr(client, "last_usage", None)
    if usage is None:
        return dict(accumulated)

    prompt = getattr(usage, "prompt_tokens", 0) or 0
    completion = getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", 0) or (prompt + completion)

    result = dict(accumulated)
    result["prompt_tokens"] = result.get("prompt_tokens", 0) + int(prompt)
    result["completion_tokens"] = result.get("completion_tokens", 0) + int(completion)
    result["total_tokens"] = result.get("total_tokens", 0) + int(total)
    return result


def reflect_and_correct_harness(
    harness: Any,
    output: HarnessOutput,
    skill_md_body: str,
    peer_outputs: dict[str, Any],
    max_rounds: int = 2,
) -> tuple[HarnessOutput, list[CorrectionItem]]:
    """Run a reflection-correction loop on a single harness output.

    Implements the Self-Refine pattern: the same LLM that produced the
    output now critiques it against three dimensions (self-consistency,
    cross-harness consistency, fidelity to source), then applies targeted
    corrections. Up to ``max_rounds`` iterations.

    Design constraints (v0.9.20):
    - **Advisory only** — never raises; on LLM failure the output is
      returned unchanged (run_harness_reflection already returns an
      all-pass sentinel on exception).
    - **Post-correction validation gate** — if the corrected result
      fails the harness's own ``validate_output``, corrections are
      rolled back to avoid propagating a broken result downstream.
    - **Reuses the harness's own client/model** — no separate large/small
      model tier; the user-configured model that produced the output also
      reviews it.
    - **Skip optimization** — high-confidence harnesses (A/F ≥ 0.9,
      E ≥ 0.95) bypass reflection entirely.

    Args:
        harness: An ``AgentHarness`` instance (provides ``client``,
            ``model``, ``name``, ``validate_output``).
        output: The ``HarnessOutput`` to reflect on. Mutated in-place
            when corrections are applied.
        skill_md_body: Original SKILL.md body text for fidelity check.
        peer_outputs: Other harness outputs (``{name: result_dict}``)
            for cross-harness consistency check.
        max_rounds: Maximum reflection iterations (default 2).

    Returns:
        ``(output, applied_corrections)``. ``applied_corrections`` is
        empty when reflection was skipped, passed clean, or rolled back.
    """
    # Skip optimization: high-confidence harnesses bypass reflection
    if should_skip_reflection(
        harness_name=harness.name,
        confidence=output.confidence,
        has_errors=not output.self_check_passed,
    ):
        output.reasoning_trace.append(
            f"[{harness.name}] reflection: skipped (confidence {output.confidence:.2f})"
        )
        return output, []

    # Pre-flight: harnesses without a validate_output override (e.g. Harness F
    # which overrides run() entirely) cannot have corrections validated —
    # skip reflection to avoid wasted LLM calls whose corrections would
    # always be rolled back.
    try:
        harness.validate_output(output.result)
    except NotImplementedError:
        output.reasoning_trace.append(
            f"[{harness.name}] reflection: skipped (no validate_output override)"
        )
        return output, []
    except Exception:
        pass  # Other validation errors are fine — reflection may fix them

    original_result = output.result
    current_result = output.result
    applied: list[CorrectionItem] = []
    last_reflection: HarnessReflectionOutput | None = None

    for round_num in range(1, max_rounds + 1):
        ctx = ReflectionContext(
            harness_output_json=json.dumps(current_result, default=str),
            skill_md_body=skill_md_body,
            peer_outputs=json.dumps(peer_outputs, default=str),
            harness_name=harness.name,
        )

        last_reflection = run_harness_reflection(
            client=harness.client,
            model=harness.model,
            ctx=ctx,
        )
        # Accumulate tokens from the reflection LLM call
        output.token_usage = _accumulate_usage(output.token_usage, harness.client)

        # Converged: all dimensions pass and no corrections pending
        all_pass = (
            last_reflection.self_consistency == "pass"
            and last_reflection.cross_harness == "pass"
            and last_reflection.fidelity == "pass"
        )
        if all_pass and not last_reflection.corrections:
            output.reasoning_trace.append(
                f"[{harness.name}] reflection: round {round_num} passed"
            )
            break

        if last_reflection.corrections:
            current_result = apply_corrections(current_result, last_reflection.corrections)
            applied.extend(last_reflection.corrections)
            output.reasoning_trace.append(
                f"[{harness.name}] reflection: round {round_num} — "
                f"{len(last_reflection.corrections)} correction(s) applied"
            )
        else:
            # Reflection flagged a failure but proposed no specific correction
            # — we cannot auto-fix, so stop iterating to avoid a no-op loop.
            output.reasoning_trace.append(
                f"[{harness.name}] reflection: round {round_num} flagged "
                f"{last_reflection.self_consistency}/{last_reflection.cross_harness}/"
                f"{last_reflection.fidelity} with no correctable field — stopping"
            )
            break

    if not applied:
        # Either skipped, passed clean, or flagged-but-unfixable.
        # Update confidence from reflection if available and lower (conservative).
        if last_reflection is not None:
            output.confidence = min(output.confidence, last_reflection.overall_confidence)
        return output, []

    # Post-correction validation gate: rollback if the corrected result
    # breaks the harness's own schema rules OR changes field types.
    try:
        passed, reason = harness.validate_output(current_result)
    except Exception as e:
        passed, reason = False, f"validate_output raised: {e}"

    # Type preservation check: corrections must not change a field's
    # Python type (e.g. list → str). This catches cases where _coerce_value
    # couldn't salvage a malformed new_value.
    if passed:
        type_changed = _detect_type_change(original_result, current_result)
        if type_changed:
            passed = False
            reason = f"correction changed field type: {type_changed}"

    if passed:
        output.result = current_result
        if last_reflection is not None:
            output.confidence = last_reflection.overall_confidence
        output.reasoning_trace.append(
            f"[{harness.name}] reflection: {len(applied)} correction(s) committed"
        )
    else:
        # Rollback: keep the original result, record the advisory
        output.result = original_result
        output.reasoning_trace.append(
            f"[{harness.name}] reflection: {len(applied)} correction(s) "
            f"rolled back (post-correction validation failed: {reason})"
        )
        applied = []  # don't claim corrections that were rolled back
        if last_reflection is not None:
            output.confidence = min(output.confidence, last_reflection.overall_confidence)

    return output, applied


def run_fidelity_checkpoint(
    client: Any,
    model: str,
    skill_md: str,
    ahspec_json: str,
) -> FidelityCheckpointOutput:
    """Run the CP2 fidelity checkpoint (Phase 2 → 3 gate).

    Verifies that the assembled AHSSpec faithfully represents the original
    SKILL.md along two axes: COVERAGE (nothing missing) and HALLUCINATION
    (nothing fabricated). Returns a fidelity score in [0.0, 1.0].

    Advisory only — never raises. On LLM failure, returns a neutral
    sentinel (score=0.5, all pass) so that the pipeline never blocks.
    """
    prompt = FIDELITY_CHECKPOINT_PROMPT.format(
        skill_md=skill_md[:4000],
        ahspec_json=ahspec_json[:4000],
    )

    try:
        result = client.chat_structured(
            messages=[{"role": "user", "content": prompt}],
            response_model=FidelityCheckpointOutput,
            model=model,
            temperature=0.1,
            thinking=True,
        )
        return result  # type: ignore[no-any-return]
    except Exception as e:
        logger.warning("Fidelity checkpoint CP2 failed: %s", e)
        return FidelityCheckpointOutput(
            coverage="pass",
            coverage_detail="CP2 LLM call failed — skipping",
            hallucination="pass",
            hallucination_detail="",
            fidelity_score=0.5,
            recommendations=[],
        )
