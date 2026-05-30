"""Phase 2: Agentic Inference Engine.

Orchestrator + AgentHarness base class + 5 concrete AgentHarness subclasses.

Architecture:
  Orchestrator (pre-flight → dispatch → collect → validate)
    ├── Harness A: extract_identity (small model)
    ├── Harness B: infer_intent (small model)
    ├── Harness C: infer_interface (large model, highest weight)
    ├── Harness D: detect_base_and_instructions (large model)
    └── Harness E: assemble_and_validate (small model)

The Orchestrator implements a 4-level error handling strategy:
  1. Transient → retry
  2. Model-recoverable → reask (Harness self-correction)
  3. User-recoverable → interrupt (Schema validation failure)
  4. Unexpected → raise (hard failure)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agenthatch.skill.llm_client import LLMClient
from agenthatch.skill.prompts import (
    ASSEMBLE_FEW_SHOT,
    ASSEMBLE_HARNESS_PERSONA,
    BASE_FEW_SHOT,
    BASE_HARNESS_PERSONA,
    FLAT_CATALOG,
    IDENTITY_FEW_SHOT,
    IDENTITY_HARNESS_PERSONA,
    INTENT_FEW_SHOT,
    INTENT_HARNESS_PERSONA,
    INTERFACE_FEW_SHOT,
    INTERFACE_HARNESS_PERSONA,
)
from agenthatch.skill.spec import (
    AHSSpec,
    BaseAndInstructionsOutput,
    BaseSpec,
    ContextPack,
    FileManifest,
    HarnessOutput,
    IdentityOutput,
    Instructions,
    IntentOutput,
    InterfaceOutput,
    _coerce_base_data,
)

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Phase 2 helpers
# ─────────────────────────────────────────────────────────────────────────


def _format_file_contents_for_harness(
    file_contents: list[dict[str, str | None]],
) -> str:
    """Format all file contents for LLM consumption.

    LLM decides which are scripts, which are docs, which are configs.
    Phase 1 makes NO semantic judgment — that's the Harness's job.
    """
    if not file_contents:
        return "(no additional files in skill directory)"

    parts: list[str] = ["\n--- Skill Directory Files ---\n"]
    for f in file_contents:
        path = f["path"]
        content = f["content"]
        if path is None:
            continue
        if content is None:
            parts.append(f"- {path} (binary or unreadable)")
            continue
        suffix = Path(path).suffix.lstrip(".") or "text"
        parts.append(f"\n### {path}\n```{suffix}\n{content}\n```\n")
    return "\n".join(parts)

# ─────────────────────────────────────────────────────────────────────────
# Model tier map (which model each Harness uses per skill type)
# ─────────────────────────────────────────────────────────────────────────

MODEL_TIER_MAP: dict[str, dict[str, str]] = {
    "pure_instruction": {"A": "small", "B": "small", "C": "large", "D": "skip", "E": "small"},
    "script_driven": {"A": "small", "B": "small", "C": "large", "D": "large", "E": "small"},
    "integration": {"A": "small", "B": "large", "C": "large", "D": "large", "E": "small"},
    "knowledge": {"A": "small", "B": "large", "C": "large", "D": "small", "E": "small"},
}


# ─────────────────────────────────────────────────────────────────────────
# AgentHarness Base Class
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class AgentHarness:
    """Base class for all 5 AgentHarnesses.

    Implements the standard Analyze → Infer → Self-Validate → Correct loop.
    Each subclass provides: persona, input schema, output schema, validation rules.
    """

    name: str
    client: LLMClient
    model: str
    max_internal_retries: int = 2

    # ── Subclass contract ──────────────────────────────────────────

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def build_user_message(self, **inputs: object) -> str:
        raise NotImplementedError

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        raise NotImplementedError

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        raise NotImplementedError

    # ── Core loop ──────────────────────────────────────────────────

    def run(self, **inputs: object) -> HarnessOutput:
        """Execute Analyze → Infer → Self-Validate → Correct loop."""
        reasoning: list[str] = []
        degradations: list[str] = []
        retries = 0

        system = self.build_system_prompt()
        user = self.build_user_message(**inputs)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

        reasoning.append(f"[{self.name}] analyze: inputs received")

        # Step 1: Initial inference
        result = self._infer(messages)
        reasoning.append(
            f"[{self.name}] infer: output received, {len(str(result))} chars"
        )

        # Step 2: Self-validate + correct loop
        while retries <= self.max_internal_retries:
            passed, reason = self.validate_output(result)
            if passed:
                reasoning.append(f"[{self.name}] self_validate: passed")
                break

            reasoning.append(f"[{self.name}] self_validate: failed — {reason}")

            if retries >= self.max_internal_retries:
                degradations.append(reason)
                reasoning.append(
                    f"[{self.name}] max retries exhausted, applying degradation"
                )
                break

            result = self.correct_on_failure(result, reason, **inputs)
            retries += 1
            reasoning.append(
                f"[{self.name}] correct: attempt {retries}/{self.max_internal_retries}"
            )

        confidence = self._estimate_confidence(result, degradations, retries)

        return HarnessOutput(
            result=result,
            confidence=confidence,
            reasoning_trace=reasoning,
            self_check_passed=len(degradations) == 0,
            degradation_applied=degradations,
            internal_retries=retries,
        )

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        """Call LLM for structured output. Override in subclasses."""
        raise NotImplementedError

    def _estimate_confidence(
        self, result: dict[str, Any], degradations: list[str], retries: int
    ) -> float:
        """Default: degrade 0.15 per retry, min 0.5."""
        base = 1.0
        base -= retries * 0.15
        if degradations:
            base *= 0.7
        return max(base, 0.5)


# ─────────────────────────────────────────────────────────────────────────
# Harness A: extract_identity
# ─────────────────────────────────────────────────────────────────────────

class ExtractIdentityHarness(AgentHarness):
    """Harness A: Extract identity fields from frontmatter + dir_name."""

    def build_system_prompt(self) -> str:
        return IDENTITY_HARNESS_PERSONA + "\n\n" + IDENTITY_FEW_SHOT

    def build_user_message(self, **inputs: object) -> str:
        frontmatter = inputs["frontmatter"]
        dir_name = inputs["dir_name"]
        body_first_50_lines = inputs["body_first_50_lines"]
        file_contents = inputs.get("file_contents", [])
        files_str = _format_file_contents_for_harness(
            file_contents if isinstance(file_contents, list) else []
        )
        return f"""Extract identity from the following skill:

dir_name: {dir_name}
frontmatter: {frontmatter}
body (first 50 lines):
{body_first_50_lines}
{files_str}"""

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        result = self.client.chat_structured(
            messages=messages,
            response_model=IdentityOutput,
            model=self.model,
        )
        return result.model_dump()

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        import re

        identity = result.get("identity", {})
        identity_id = identity.get("id", "")
        if not identity_id:
            return False, "identity.id is empty"
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", identity_id):
            return False, f"identity.id '{identity_id}' is not kebab-case"
        if not identity.get("display_name"):
            return False, "identity.display_name is empty"
        if not identity.get("version"):
            return False, "identity.version is empty"
        return True, ""

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        frontmatter = inputs["frontmatter"]
        dir_name = inputs["dir_name"]
        body_first_50_lines = inputs["body_first_50_lines"]
        correction_prompt = (
            f"Your previous output failed validation: {failure_reason}\n\n"
            "Please fix and return corrected identity:\n"
            f"{self.build_user_message(frontmatter=frontmatter, dir_name=dir_name, body_first_50_lines=str(body_first_50_lines)[:500])}"  # noqa: E501
        )
        corrected_result = self.client.chat_structured(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": correction_prompt},
            ],
            response_model=IdentityOutput,
            model=self.model,
        )
        return cast("dict[str, Any]", corrected_result.model_dump())


# ─────────────────────────────────────────────────────────────────────────
# Harness B: infer_intent
# ─────────────────────────────────────────────────────────────────────────

class InferIntentHarness(AgentHarness):
    """Harness B: Infer intent — triggers, satisfies, summary."""

    def build_system_prompt(self) -> str:
        return INTENT_HARNESS_PERSONA + "\n\n" + INTENT_FEW_SHOT

    def build_user_message(self, **inputs: object) -> str:
        description = inputs["description"]
        body = inputs["body"]
        frontmatter_name = inputs["frontmatter_name"]
        file_contents = inputs.get("file_contents", [])
        desc = description or "(not provided)"
        name = frontmatter_name or "(not provided)"
        body_preview = str(body)[:3000]
        files_str = _format_file_contents_for_harness(
            file_contents if isinstance(file_contents, list) else []
        )
        return f"""Infer intent for this skill:

description: {desc}
frontmatter_name: {name}
body:
{body_preview}
{files_str}"""

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        result = self.client.chat_structured(
            messages=messages,
            response_model=IntentOutput,
            model=self.model,
        )
        return result.model_dump()

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        intent = result.get("intent", {})
        triggers = intent.get("triggers", [])
        satisfies = intent.get("satisfies", [])
        summary = intent.get("summary", "")

        if not (5 <= len(triggers) <= 15):
            return False, f"triggers count {len(triggers)} not in [5, 15]"
        if not (3 <= len(satisfies) <= 8):
            return False, f"satisfies count {len(satisfies)} not in [3, 8]"
        if len(summary) < 20:
            return False, f"summary too short ({len(summary)} chars, need >= 20)"
        return True, ""

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        description = inputs["description"]
        body = inputs["body"]
        frontmatter_name = inputs["frontmatter_name"]
        correction_prompt = (
            f"Your previous output failed validation: {failure_reason}\n\n"
            "Please fix and return corrected intent:\n"
            f"{self.build_user_message(description=description, body=str(body)[:2000], frontmatter_name=frontmatter_name)}"  # noqa: E501
        )
        corrected_result = self.client.chat_structured(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": correction_prompt},
            ],
            response_model=IntentOutput,
            model=self.model,
        )
        return cast("dict[str, Any]", corrected_result.model_dump())


# ─────────────────────────────────────────────────────────────────────────
# Harness C: infer_interface
# ─────────────────────────────────────────────────────────────────────────

class InferInterfaceHarness(AgentHarness):
    """Harness C: Infer capability interface — provides, requires, compatible_with."""

    def build_system_prompt(self) -> str:
        catalog_str = "\n".join(
            f"  - {cap}" for cap in sorted(FLAT_CATALOG)
        )
        return (
            INTERFACE_HARNESS_PERSONA
            + f"\n\nAvailable infrastructure capabilities:\n{catalog_str}\n\n"
            + INTERFACE_FEW_SHOT
        )

    def build_user_message(self, **inputs: object) -> str:
        body = inputs["body"]
        file_contents = inputs["file_contents"]
        frontmatter_allowed_tools = inputs["frontmatter_allowed_tools"]
        files_text = _format_file_contents_for_harness(
            file_contents if isinstance(file_contents, list) else []
        )
        tools = frontmatter_allowed_tools if isinstance(frontmatter_allowed_tools, list) else []
        return f"""Infer the interface for this skill:

frontmatter_allowed_tools: {tools}
body:
{str(body)[:4000]}

{files_text}"""

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        result = self.client.chat_structured(
            messages=messages,
            response_model=InterfaceOutput,
            model=self.model,
        )
        return result.model_dump()

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        interface = result.get("interface", {})
        provides = interface.get("provides", [])

        if not provides:
            return False, "interface.provides is empty — fatal"

        # Check capability names are snake_case
        import re
        for cap in provides:
            name = cap.get("capability", "")
            if not re.match(r"^[a-z][a-z0-9]*(_[a-z0-9]+)*$", name):
                return False, f"capability '{name}' is not snake_case"

        # Check requires are in catalog
        for req in interface.get("requires", []):
            cap_name = req.get("capability", "")
            if cap_name not in FLAT_CATALOG and not req.get("optional"):
                return False, (
                    f"requirement '{cap_name}' not in infrastructure catalog "
                    f"and not marked optional"
                )

        return True, ""

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        body = inputs["body"]
        file_contents = inputs["file_contents"]
        frontmatter_allowed_tools = inputs["frontmatter_allowed_tools"]
        correction_prompt = (
            f"Your previous output failed validation: {failure_reason}\n\n"
            "Please fix and return corrected interface:\n"
            f"{self.build_user_message(body=str(body)[:2000], file_contents=file_contents, frontmatter_allowed_tools=frontmatter_allowed_tools)}"  # noqa: E501
        )
        corrected_result = self.client.chat_structured(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": correction_prompt},
            ],
            response_model=InterfaceOutput,
            model=self.model,
        )
        return cast("dict[str, Any]", corrected_result.model_dump())


# ─────────────────────────────────────────────────────────────────────────
# Harness D: detect_base_and_instructions
# ─────────────────────────────────────────────────────────────────────────

class DetectBaseHarness(AgentHarness):
    """Harness D: Detect runtime base and instruction structure."""

    def build_system_prompt(self) -> str:
        return BASE_HARNESS_PERSONA + "\n\n" + BASE_FEW_SHOT

    def build_user_message(self, **inputs: object) -> str:
        body = inputs["body"]
        file_contents = inputs["file_contents"]
        frontmatter_compatibility = inputs["frontmatter_compatibility"]
        frontmatter_allowed_tools = inputs["frontmatter_allowed_tools"]
        files_text = _format_file_contents_for_harness(
            file_contents if isinstance(file_contents, list) else []
        )
        compat = frontmatter_compatibility or "(not provided)"
        tools = frontmatter_allowed_tools if isinstance(frontmatter_allowed_tools, list) else []
        return f"""Detect base and instructions for this skill:

frontmatter_compatibility: {compat}
frontmatter_allowed_tools: {tools}
body:
{str(body)[:4000]}

{files_text}"""

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        result = self.client.chat_structured(
            messages=messages,
            response_model=BaseAndInstructionsOutput,
            model=self.model,
        )
        return result.model_dump()

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        valid_runtimes = {"python3.11", "bash", "node20", None}
        runtime = result.get("base", {}).get("runtime")
        instructions = result.get("instructions", {})

        if runtime is not None and runtime not in valid_runtimes:
            return False, f"Invalid runtime: {runtime} (expected one of {valid_runtimes})"
        if not instructions.get("workflow"):
            return False, "instructions.workflow is empty"
        return True, ""

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        body = inputs["body"]
        file_contents = inputs["file_contents"]
        frontmatter_compatibility = inputs["frontmatter_compatibility"]
        frontmatter_allowed_tools = inputs["frontmatter_allowed_tools"]
        correction_prompt = (
            f"Your previous output failed validation: {failure_reason}\n\n"
            "Please fix and return corrected base and instructions:\n"
            f"{self.build_user_message(body=str(body)[:2000], file_contents=file_contents, frontmatter_compatibility=frontmatter_compatibility, frontmatter_allowed_tools=frontmatter_allowed_tools)}"  # noqa: E501
        )
        corrected_result = self.client.chat_structured(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": correction_prompt},
            ],
            response_model=BaseAndInstructionsOutput,
            model=self.model,
        )
        return cast("dict[str, Any]", corrected_result.model_dump())


# ─────────────────────────────────────────────────────────────────────────
# Harness E: assemble_and_validate
# ─────────────────────────────────────────────────────────────────────────

class AssembleHarness(AgentHarness):
    """Harness E: Final assembly and cross-validation."""

    def build_system_prompt(self) -> str:
        return ASSEMBLE_HARNESS_PERSONA + "\n\n" + ASSEMBLE_FEW_SHOT

    def build_user_message(self, **inputs: object) -> str:
        identity = inputs["identity"]
        intent = inputs["intent"]
        interface = inputs["interface"]
        base = inputs["base"]
        instructions = inputs["instructions"]
        resources = inputs["resources"]
        dir_name = inputs["dir_name"]
        import json

        return f"""Assemble the final AHSSPEC from these harness outputs:

identity: {json.dumps(identity, ensure_ascii=False, indent=2)}
intent: {json.dumps(intent, ensure_ascii=False, indent=2)}
interface: {json.dumps(interface, ensure_ascii=False, indent=2)}
base: {json.dumps(base, ensure_ascii=False, indent=2)}
instructions: {json.dumps(instructions, ensure_ascii=False, indent=2)}
resources: {json.dumps(resources, ensure_ascii=False, indent=2)}
dir_name: {dir_name}

Cross-check and return the assembled ahs_spec with confidence_report."""

    def _infer(self, messages: list[dict[str, Any]]) -> Any:
        """Harness E uses simple chat (not structured) for flexible assembly."""
        response = self.client.chat(
            messages=messages,
            model=self.model,
            temperature=0.3,
            max_tokens=8192,
        )
        import json

        # Extract JSON from response (may be wrapped in markdown code blocks)
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        return json.loads(text)

    def validate_output(self, result: dict[str, Any]) -> tuple[bool, str]:
        if not result.get("ahs_spec"):
            return False, "ahs_spec is missing"
        if "identity" not in result.get("ahs_spec", {}):
            return False, "ahs_spec.identity is missing"
        if "interface" not in result.get("ahs_spec", {}):
            return False, "ahs_spec.interface is missing"
        if not result.get("ahs_spec", {}).get("interface", {}).get("provides"):
            return False, "ahs_spec.interface.provides is empty — fatal"
        return True, ""

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        identity = inputs["identity"]
        intent = inputs["intent"]
        interface = inputs["interface"]
        base = inputs["base"]
        instructions = inputs["instructions"]
        resources = inputs["resources"]
        dir_name = inputs["dir_name"]
        correction_prompt = (
            f"Your previous output failed validation: {failure_reason}\n\n"
            "Please fix and return the corrected ahs_spec:\n"
            f"{self.build_user_message(identity=identity, intent=intent, interface=interface, base=base, instructions=instructions, resources=resources, dir_name=dir_name)}"  # noqa: E501
        )
        response = self.client.chat(
            messages=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": correction_prompt},
            ],
            model=self.model,
            temperature=0.3,
            max_tokens=8192,
        )
        import json

        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            return cast("dict[str, Any]", json.loads(text))
        except json.JSONDecodeError as e:
            logger.warning(f"Harness E correction JSON parse failed: {e}")
            return result


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator

class Orchestrator:
    """Phase 2 orchestration agent.

    Responsibilities:
      1. Pre-flight: analyze skill type, decide Harness combination + model tiers
      2. Build: construct LLMClient + 5 AgentHarness instances
      3. Dispatch: parallel A+B+C, sequential D, sequential E
      4. Collect: gather HarnessOutput, check self_check_passed
      5. Finalize: if validation fails, enter targeted repair loop
    """

    def __init__(self, config: dict[str, Any], large_model: str = "", small_model: str = ""):
        """Initialize Orchestrator from config.

        Args:
            config: Full config dict from Config.load().
            large_model: Override for large model tier (empty = use default).
            small_model: Override for small model tier (empty = use default).
        """
        from agenthatch.providers import get_provider

        provider_name = config.get("providers", {}).get("default", "openai")
        provider_info = get_provider(provider_name)
        default_model = provider_info.default_model

        self._large_model = large_model or default_model
        self._small_model = small_model or default_model

        self._large_client = LLMClient(provider_name=provider_name, model=self._large_model)
        self._small_client = LLMClient(provider_name=provider_name, model=self._small_model)

        self._provider_name = provider_name

    def run(self, context: ContextPack) -> tuple[AHSSpec, dict[str, HarnessOutput]]:
        """Run Phase 2 on a ContextPack.

        Returns:
            (validated AHSSpec, harness_outputs dict).
        """
        # Step 0: Pre-flight classification
        skill_type = self._classify(context)
        tier_map = MODEL_TIER_MAP.get(skill_type, MODEL_TIER_MAP["pure_instruction"])
        logger.info(f"Orchestrator: skill_type={skill_type}, tiers={tier_map}")

        # Step 1: Build harness instances
        harnesses = self._build_harnesses(tier_map)

        # Phase 2 generates resources deterministically (not via LLM)
        resources = self._build_resources(context.file_manifest)

        # Extract all readable file contents (flat, no classification)
        file_contents = context.file_manifest.content_bundle()

        # Step 2: Parallel dispatch (A + B + C)
        outputs: dict[str, HarnessOutput] = {}

        if tier_map.get("A") != "skip":
            outputs["A"] = harnesses["A"].run(
                frontmatter=context.frontmatter,
                dir_name=context.dir_name,
                body_first_50_lines=context.body[:2500],
                file_contents=file_contents,
            )

        if tier_map.get("B") != "skip":
            outputs["B"] = harnesses["B"].run(
                description=context.frontmatter.get("description") if context.frontmatter else None,
                body=context.body,
                frontmatter_name=context.frontmatter.get("name") if context.frontmatter else None,
                file_contents=file_contents,
            )

        if tier_map.get("C") != "skip":
            outputs["C"] = harnesses["C"].run(
                body=context.body,
                file_contents=file_contents,
                frontmatter_allowed_tools=(
                    context.frontmatter.get("allowed_tools") if context.frontmatter else None
                ),
            )

        # Step 3: Check self-validation
        for name, output in outputs.items():
            if not output.self_check_passed:
                logger.warning(
                    f"Harness {name} self_check failed: {output.degradation_applied}"
                )

        # Step 4: Sequential dispatch (D depends on C for runtime context)
        if tier_map.get("D") != "skip":
            outputs["D"] = harnesses["D"].run(
                body=context.body,
                file_contents=file_contents,
                frontmatter_compatibility=(
                    context.frontmatter.get("compatibility") if context.frontmatter else None
                ),
                frontmatter_allowed_tools=(
                    context.frontmatter.get("allowed_tools") if context.frontmatter else None
                ),
            )

        # Step 5: Assemble (E)
        try:
            outputs["E"] = harnesses["E"].run(
                identity=outputs["A"].result if "A" in outputs else {},
                intent=outputs["B"].result if "B" in outputs else {},
                interface=outputs["C"].result if "C" in outputs else {},
                base=outputs["D"].result.get("base", {}) if "D" in outputs else {},
                instructions=outputs["D"].result.get("instructions", {}) if "D" in outputs else {},
                resources=resources,
                dir_name=context.dir_name,
            )
        except Exception as e:
            logger.error(f"Harness E assembly failed: {e}")
            from agenthatch.skill.validate import validate_and_repair
            return validate_and_repair({}, outputs, harnesses, context)

        # Step 6: Build AHSSpec from Harness E assembly output
        try:
            ahs_dict = outputs["E"].result.get("ahs_spec", {})
            ahs_spec = self._dict_to_ahspec(ahs_dict)

            # Attach confidence report and traces
            confidence_report = outputs["E"].result.get("confidence_report", {})
            from agenthatch.skill.spec import ConfidenceReport

            if confidence_report:
                ahs_spec.confidence_report = ConfidenceReport(**confidence_report)
            ahs_spec.harness_traces = [outputs[k] for k in ["A", "B", "C", "D", "E"] if k in outputs]  # noqa: E501

            return ahs_spec, outputs
        except Exception as e:
            # Last resort: try targeted repair via Pydantic validation
            logger.warning(f"Assembly failed: {e}, attempting targeted repair")
            from agenthatch.skill.validate import validate_and_repair

            return validate_and_repair(outputs["E"].result, outputs, harnesses, context)

    def _classify(self, context: ContextPack) -> str:
        """Pre-flight skill type classification (deterministic heuristics).

        Uses flat FileManifest (DD-E01: Phase 1 makes no semantic judgment,
        but Phase 2 pre-flight can use basic heuristics for routing).
        """
        _SCRIPT_SUFFIXES = {".py", ".sh", ".js", ".ts", ".rb", ".go", ".rs"}
        entries = context.file_manifest.entries
        has_scripts = any(
            Path(e.path).suffix.lower() in _SCRIPT_SUFFIXES for e in entries
        )
        body_lower = context.body.lower()

        api_indicators = ["api", "oauth", "token", "http", "rest", "webhook"]
        has_api = any(ind in body_lower for ind in api_indicators)

        if has_scripts and has_api:
            return "integration"
        if has_scripts:
            return "script_driven"
        if len(entries) > 2:
            return "knowledge"
        return "pure_instruction"

    def _build_harnesses(self, tier_map: dict[str, str]) -> dict[str, AgentHarness]:
        """Build AgentHarness instances with tier-appropriate models/clients."""

        def _resolve(harness_id: str) -> tuple[LLMClient, str]:
            tier = tier_map.get(harness_id, "small")
            if tier == "large":
                return self._large_client, self._large_model
            return self._small_client, self._small_model

        a_client, a_model = _resolve("A")
        b_client, b_model = _resolve("B")
        c_client, c_model = _resolve("C")
        d_client, d_model = _resolve("D")
        e_client, e_model = _resolve("E")

        return {
            "A": ExtractIdentityHarness(name="extract_identity", client=a_client, model=a_model),
            "B": InferIntentHarness(name="infer_intent", client=b_client, model=b_model),
            "C": InferInterfaceHarness(name="infer_interface", client=c_client, model=c_model),
            "D": DetectBaseHarness(name="detect_base_and_instructions", client=d_client, model=d_model),  # noqa: E501
            "E": AssembleHarness(name="assemble_and_validate", client=e_client, model=e_model),
        }

    def _build_resources(self, manifest: FileManifest) -> dict[str, list[dict[str, str]]]:
        """Build resources dict from flat file manifest (deterministic).

        All non-entrypoint files with content are listed as resources.
        LLM determines semantic role — Phase 2 heuristic only groups by extension.
        """
        _SCRIPT_SUFFIXES = {".py", ".sh", ".js", ".ts", ".rb", ".go", ".rs"}
        _REF_SUFFIXES = {".md", ".rst", ".txt", ".yaml", ".yml", ".json", ".toml"}

        scripts: list[dict[str, str]] = []
        references: list[dict[str, str]] = []
        assets: list[dict[str, str]] = []

        for e in manifest.entries:
            if e.path == manifest.entrypoint:
                continue  # Skip SKILL.md itself
            suffix = Path(e.path).suffix.lower()
            entry_dict = {"name": e.path, "hash": e.hash}
            if suffix in _SCRIPT_SUFFIXES:
                scripts.append(entry_dict)
            elif suffix in _REF_SUFFIXES:
                references.append(entry_dict)
            else:
                assets.append(entry_dict)

        return {"scripts": scripts, "references": references, "assets": assets}

    def _dict_to_ahspec(self, ahs_dict: dict[str, Any]) -> AHSSpec:
        """Convert raw dict from Harness E into validated AHSSpec."""
        from agenthatch.skill.spec import (
            Composition,
            Identity,
            Intent,
            Interface,
        )

        identity = Identity(**ahs_dict.get("identity", {}))
        intent = Intent(**ahs_dict.get("intent", {}))
        interface = Interface(**ahs_dict.get("interface", {}))
        base = BaseSpec(**_coerce_base_data(ahs_dict.get("base", {})))
        instructions = Instructions(**ahs_dict.get("instructions", {}))
        composition = Composition(**ahs_dict.get("composition", {})) if ahs_dict.get("composition") else Composition()  # noqa: E501

        return AHSSpec(
            identity=identity,
            intent=intent,
            interface=interface,
            base=base,
            instructions=instructions,
            composition=composition,
        )
