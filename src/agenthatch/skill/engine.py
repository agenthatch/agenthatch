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

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from agenthatch_core.llm.client import LLMClient
from pydantic import ValidationError

from agenthatch.skill.prompts import (
    ASSEMBLE_FEW_SHOT,
    ASSEMBLE_HARNESS_PERSONA,
    BASE_FEW_SHOT,
    BASE_HARNESS_PERSONA,
    FLAT_CATALOG,
    IDENTITY_FEW_SHOT,
    IDENTITY_HARNESS_PERSONA,
    INFER_MCP_SERVERS_PROMPT,
    INTENT_FEW_SHOT,
    INTENT_HARNESS_PERSONA,
    INTERFACE_FEW_SHOT,
    INTERFACE_HARNESS_PERSONA,
)
from agenthatch.skill.spec import (
    AgentConfig,
    AgentRuntimeConfig,
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
    Resources,
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
    "pure_instruction": {
        "A": "small", "B": "small", "C": "large", "D": "skip", "E": "small", "F": "small",
    },
    "script_driven": {
        "A": "small", "B": "small", "C": "large", "D": "large", "E": "small", "F": "small",
    },
    "integration": {
        "A": "small", "B": "large", "C": "large", "D": "large", "E": "large", "F": "small",
    },
    "knowledge": {
        "A": "small", "B": "large", "C": "large", "D": "small", "E": "small", "F": "small",
    },
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

    # ── Correction hooks (Template Method) ────────────────────────

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        """Subclass overrides to extract/preprocess inputs for correction."""
        return inputs

    def _get_correction_response_model(self) -> Any:
        """Subclass overrides to return the Pydantic model for structured output."""
        return None

    def _use_structured_output_for_correction(self) -> bool:
        """Subclass overrides to return False for raw chat (e.g. AssembleHarness)."""
        return True

    def _parse_correction_response(
        self, response: Any, result: dict[str, Any]
    ) -> dict[str, Any]:
        """Subclass overrides to customize response parsing."""
        if hasattr(response, "model_dump"):
            return cast("dict[str, Any]", response.model_dump())
        return cast("dict[str, Any]", response)

    def _get_correction_output_type_name(self) -> str:
        """Subclass overrides to return a human-readable output type name."""
        return "output"

    def _get_correction_kwargs(self) -> dict[str, Any]:
        """Subclass overrides to pass extra kwargs to LLM call (e.g. temperature)."""
        return {}

    def _build_correction_prompt(
        self, result: dict[str, Any], failure_reason: str, prepared: dict[str, Any]
    ) -> str:
        """Subclass overrides to customize the correction prompt."""
        output_type = self._get_correction_output_type_name()
        return (
            f"Your previous output failed validation: {failure_reason}\n\n"
            f"Please fix and return corrected {output_type}:\n"
            f"{self.build_user_message(**prepared)}"
        )

    def correct_on_failure(
        self, result: dict[str, Any], failure_reason: str, **inputs: object
    ) -> dict[str, Any]:
        """Unified correction loop via Template Method pattern.

        Subclasses provide the variation points via hooks:
        _prepare_correction_inputs, _get_correction_response_model,
        _use_structured_output_for_correction, _parse_correction_response,
        _get_correction_output_type_name, _build_correction_prompt,
        _get_correction_kwargs.
        """
        prepared = self._prepare_correction_inputs(**inputs)
        correction_prompt = self._build_correction_prompt(result, failure_reason, prepared)

        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.build_system_prompt()},
            {"role": "user", "content": correction_prompt},
        ]

        extra_kwargs = self._get_correction_kwargs()

        if self._use_structured_output_for_correction():
            response_model = self._get_correction_response_model()
            if response_model is None:
                raise NotImplementedError(
                    f"{self.__class__.__name__} must override "
                    "_get_correction_response_model() or "
                    "_use_structured_output_for_correction()"
                )
            corrected = self.client.chat_structured(
                messages=messages,
                response_model=response_model,
                model=self.model,
                **extra_kwargs,
            )
        else:
            corrected = self.client.chat(
                messages=messages,
                model=self.model,
                **extra_kwargs,
            )

        return self._parse_correction_response(corrected, result)

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

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        return {
            "frontmatter": inputs["frontmatter"],
            "dir_name": inputs["dir_name"],
            "body_first_50_lines": str(inputs["body_first_50_lines"])[:500],
        }

    def _get_correction_response_model(self) -> Any:
        return IdentityOutput

    def _get_correction_output_type_name(self) -> str:
        return "identity"


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

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        return {
            "description": inputs["description"],
            "body": str(inputs["body"])[:2000],
            "frontmatter_name": inputs["frontmatter_name"],
        }

    def _get_correction_response_model(self) -> Any:
        return IntentOutput

    def _get_correction_output_type_name(self) -> str:
        return "intent"


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

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        return {
            "body": str(inputs["body"])[:2000],
            "file_contents": inputs["file_contents"],
            "frontmatter_allowed_tools": inputs["frontmatter_allowed_tools"],
        }

    def _get_correction_response_model(self) -> Any:
        return InterfaceOutput

    def _get_correction_output_type_name(self) -> str:
        return "interface"


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

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        return {
            "body": str(inputs["body"])[:2000],
            "file_contents": inputs["file_contents"],
            "frontmatter_compatibility": inputs["frontmatter_compatibility"],
            "frontmatter_allowed_tools": inputs["frontmatter_allowed_tools"],
        }

    def _get_correction_response_model(self) -> Any:
        return BaseAndInstructionsOutput

    def _get_correction_output_type_name(self) -> str:
        return "base and instructions"


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
        """Harness E: try structured output first, fallback to raw chat."""
        # ── v0.5.10: Prefer chat_structured for reliability ──
        try:
            from agenthatch.skill.spec import AssembleOutput
            result = self.client.chat_structured(
                messages=messages,
                response_model=AssembleOutput,
                model=self.model,
                temperature=0.3,
                max_tokens=8192,
            )
            return result.model_dump()
        except Exception as e:
            logger.debug(f"Harness E chat_structured failed: {e}, falling back to raw chat")

        # Fallback: raw chat with manual JSON extraction
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
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return {}

    def _compute_structural_confidence(self, ahs_dict: dict[str, Any]) -> float:
        """Compute confidence based on structural checks, not LLM self-assessment."""
        checks = 0
        passed = 0

        id_ = ahs_dict.get("identity", {})
        for f in ("id", "display_name", "version"):
            checks += 1
            if id_.get(f):
                passed += 1

        iface = ahs_dict.get("interface", {})
        for f in ("provides", "requires"):
            checks += 1
            if iface.get(f):
                passed += 1

        instr = ahs_dict.get("instructions", {})
        for f in ("workflow", "rules"):
            checks += 1
            if instr.get(f):
                passed += 1

        res = ahs_dict.get("resources", {})
        checks += 1
        if res.get("scripts") or res.get("references"):
            passed += 1

        score = round(passed / max(checks, 1), 2)
        logger.info("Harness E structural confidence: %.2f (%d/%d)", score, passed, checks)
        return score

    def run(self, **inputs: object) -> HarnessOutput:
        output = super().run(**inputs)
        ahs_dict = output.result.get("ahs_spec", {})
        structural_confidence = self._compute_structural_confidence(ahs_dict)
        output.confidence = structural_confidence
        return output

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

    def _prepare_correction_inputs(self, **inputs: object) -> dict[str, Any]:
        return {
            "identity": inputs["identity"],
            "intent": inputs["intent"],
            "interface": inputs["interface"],
            "base": inputs["base"],
            "instructions": inputs["instructions"],
            "resources": inputs["resources"],
            "dir_name": inputs["dir_name"],
        }

    def _use_structured_output_for_correction(self) -> bool:
        return False

    def _get_correction_output_type_name(self) -> str:
        return "ahs_spec"

    def _build_correction_prompt(
        self, result: dict[str, Any], failure_reason: str, prepared: dict[str, Any]
    ) -> str:
        return (
            f"Your previous output failed validation: {failure_reason}\n\n"
            f"Previous output (first 500 chars):\n{str(result)[:500]}\n\n"
            "Please fix and return the corrected ahs_spec:\n"
            f"{self.build_user_message(**prepared)}"
        )

    def _get_correction_kwargs(self) -> dict[str, Any]:
        return {"temperature": 0.3, "max_tokens": 8192}

    def _parse_correction_response(
        self, response: str, result: dict[str, Any]
    ) -> dict[str, Any]:
        text = response.strip()
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0].strip()
        elif "```" in text:
            text = text.split("```")[1].split("```")[0].strip()
        try:
            import json
            return cast("dict[str, Any]", json.loads(text))
        except json.JSONDecodeError as e:
            logger.warning(f"Harness E correction JSON parse failed: {e}")
            return result


# ─────────────────────────────────────────────────────────────────────────
# Harness F: infer_mcp_servers
# ─────────────────────────────────────────────────────────────────────────

class InferMCPServersHarness(AgentHarness):
    """Harness F: Detect MCP server dependencies from skill body."""

    def build_system_prompt(self) -> str:
        return INFER_MCP_SERVERS_PROMPT

    def build_user_message(self, **inputs: object) -> str:
        body = str(inputs.get("body", ""))
        return body

    def run(self, **inputs: object) -> HarnessOutput:
        body = str(inputs.get("body", ""))
        mcp_servers: list[dict[str, Any]] = []
        if not body:
            return HarnessOutput(
                result={"mcp_servers": mcp_servers},
                confidence=1.0,
                reasoning_trace=["no body, skipping"],
                self_check_passed=True,
            )

        try:
            messages = [
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": self.build_user_message(body=body)},
            ]
            response = self.client.chat(messages=messages, model=self.model)
            import json as _json

            data = _json.loads(response)
            llm_servers = data.get("mcp_servers", [])
        except Exception:
            # Fallback to regex-based extraction
            llm_servers = []
            mcp_pattern = re.compile(r'mcp__(\w+)__')
            server_names = set(mcp_pattern.findall(body))
            for sname in sorted(server_names):
                llm_servers.append({
                    "name": sname,
                    "config": {"transport": "auto"},
                })

        # Verify against actual mcp__ patterns in body
        mcp_pattern = re.compile(r'mcp__([a-zA-Z0-9_-]+)__')
        referenced = set(mcp_pattern.findall(body))
        for server in llm_servers:
            name = server.get("name", "")
            if name in referenced:
                mcp_servers.append(server)
            else:
                logger.warning(
                    "Harness F: dropping MCP server '%s' — not referenced in SKILL.md", name
                )

        return HarnessOutput(
            result={"mcp_servers": mcp_servers},
            confidence=0.9 if mcp_servers else 0.5,
            reasoning_trace=[f"detected {len(mcp_servers)} MCP servers"],
            self_check_passed=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator

class Orchestrator:
    """Phase 2 orchestration agent.

    Responsibilities:
      1. Pre-flight: analyze skill type, decide Harness combination + model tiers
      2. Build: construct LLMClient + 5 AgentHarness instances
      3. Dispatch: sequential A → B → C → D → E
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

        self._large_client = LLMClient(provider=provider_name, model=self._large_model)
        self._small_client = LLMClient(provider=provider_name, model=self._small_model)

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
            logger.info("Running harness A: extract_identity")
            outputs["A"] = harnesses["A"].run(
                frontmatter=context.frontmatter,
                dir_name=context.dir_name,
                body_first_50_lines=context.body[:2500],
                file_contents=file_contents,
            )

        if tier_map.get("B") != "skip":
            logger.info("Running harness B: infer_intent")
            outputs["B"] = harnesses["B"].run(
                description=context.frontmatter.get("description") if context.frontmatter else None,
                body=context.body,
                frontmatter_name=context.frontmatter.get("name") if context.frontmatter else None,
                file_contents=file_contents,
            )

        if tier_map.get("C") != "skip":
            logger.info("Running harness C: infer_interface")
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
            logger.info("Running harness D: detect_base_and_instructions")
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
            logger.info("Running harness E: assemble_and_validate")
            outputs["E"] = harnesses["E"].run(
                identity=outputs["A"].result if "A" in outputs else {},
                intent=outputs["B"].result if "B" in outputs else {},
                interface=outputs["C"].result if "C" in outputs else {},
                base=outputs["D"].result.get("base", {}) if "D" in outputs else {},
                instructions=outputs["D"].result.get("instructions", {}) if "D" in outputs else {},
                resources=resources,
                dir_name=context.dir_name,
            )
        except (ValueError, TypeError, RuntimeError, json.JSONDecodeError) as e:
            logger.warning(f"Harness E assembly failed: {e}, retrying once")
            try:
                outputs["E"] = harnesses["E"].run(
                    identity=outputs["A"].result if "A" in outputs else {},
                    intent=outputs["B"].result if "B" in outputs else {},
                    interface=outputs["C"].result if "C" in outputs else {},
                    base=outputs["D"].result.get("base", {}) if "D" in outputs else {},
                    instructions=(
                        outputs["D"].result.get("instructions", {})
                        if "D" in outputs else {}
                    ),
                    resources=resources,
                    dir_name=context.dir_name,
                )
            except Exception as e2:
                logger.error(f"Harness E retry also failed: {e2}")
                from agenthatch.exceptions import SchemaValidationError
                raise SchemaValidationError(f"Harness E failed: {e2}") from e2

        # Step 5b: Infer MCP servers (F)
        if tier_map.get("F") != "skip":
            logger.info("Running harness F: infer_mcp_servers")
            outputs["F"] = harnesses["F"].run(
                body=context.body,
                references=resources.get("references", []),
                api_templates=None,
            )

        # Step 6: Build AHSSpec from Harness E assembly output
        ahs_dict: dict[str, Any] = {}
        try:
            ahs_dict = outputs["E"].result.get("ahs_spec", {})

            # Wire resources into ahs_dict
            ahs_dict["resources"] = resources

            # Inject raw_body into instructions
            if "instructions" not in ahs_dict:
                ahs_dict["instructions"] = {}
            ahs_dict["instructions"]["raw_body"] = context.body

            # API template detection
            api_templates = self._detect_api_templates(context.body)
            if "interface" not in ahs_dict:
                ahs_dict["interface"] = {}
            ahs_dict["interface"]["api_templates"] = api_templates

            # Merge MCP servers from Harness F
            mcp_servers: list[dict[str, Any]] = []
            if "F" in outputs:
                f_output: Any = outputs["F"]
                mcp_servers = (
                    f_output.result.get("mcp_servers", [])
                    if hasattr(f_output, "result") else []
                )
            # Enrich MCP servers from SKILL.md body patterns
            mcp_servers = self._enrich_mcp_from_body(mcp_servers, context.body)
            if mcp_servers:
                if "interface" not in ahs_dict:
                    ahs_dict["interface"] = {}
                ahs_dict["interface"]["mcp_servers"] = mcp_servers

            ahs_spec = self._dict_to_ahspec(ahs_dict)

            # Attach confidence report and traces
            confidence_report = outputs["E"].result.get("confidence_report", {})
            from agenthatch.skill.spec import ConfidenceReport

            if confidence_report:
                ahs_spec.confidence_report = ConfidenceReport(**confidence_report)
            ahs_spec.harness_traces = [outputs[k] for k in ["A", "B", "C", "D", "E", "F"] if k in outputs]  # noqa: E501

            return ahs_spec, outputs
        except (ValidationError, TypeError, ValueError) as e:
            logger.warning(f"Assembly failed: {e}, attempting targeted repair")
            from agenthatch.skill.validate import validate_and_repair

            return validate_and_repair(ahs_dict, outputs, harnesses, context)

    def _classify(self, context: ContextPack) -> str:
        """Pre-flight skill type classification (deterministic heuristics).

        Uses flat FileManifest (Phase 1 makes no semantic judgment,
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
        f_client, f_model = _resolve("F")

        # Extend timeout for reasoning models
        d_timeout = 60
        if d_client and hasattr(d_client, '_features'):
            if d_client._features.supports_reasoning_content:
                d_timeout = 120
        logger.debug("Harness D timeout: %ds", d_timeout)

        return {
            "A": ExtractIdentityHarness(name="extract_identity", client=a_client, model=a_model),
            "B": InferIntentHarness(name="infer_intent", client=b_client, model=b_model),
            "C": InferInterfaceHarness(name="infer_interface", client=c_client, model=c_model),
            "D": DetectBaseHarness(name="detect_base_and_instructions", client=d_client, model=d_model),  # noqa: E501
            "E": AssembleHarness(name="assemble_and_validate", client=e_client, model=e_model),
            "F": InferMCPServersHarness(name="infer_mcp_servers", client=f_client, model=f_model),
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

    _CURL_PATTERN = re.compile(
        r'curl\s+(?:-[a-zA-Z]+\s+)*'
        r'(?:["\'])?(https?://[^\s"\']+)(?:["\'])?'
    )

    @staticmethod
    def _derive_api_name(url: str, method: str) -> str:
        """Derive a human-readable name from URL path for API template."""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.strip("/").split("/") if p]
        if len(path_parts) >= 3:
            name = "_".join(path_parts[-3:])
        elif path_parts:
            name = "_".join(path_parts[-2:])
        else:
            name = parsed.netloc.replace(".", "_")
        # Clean up non-alphanumeric
        import re as _re
        name = _re.sub(r'[^a-zA-Z0-9_]', '_', name)
        return f"{method.lower()}_{name}"[:50]

    @staticmethod
    def _enrich_mcp_from_body(
        servers: list[dict[str, Any]], body: str
    ) -> list[dict[str, Any]]:
        """Scan SKILL.md body for mcp__ patterns.

        Fills in missing server info from mcp__SERVER__TOOL references.
        Never fabricates URLs — if no URL in SKILL.md, transport stays empty.
        """
        mcp_pattern = re.compile(r'mcp__(\w[\w-]*)__')
        found = set(mcp_pattern.findall(body))
        existing_names = {s.get('name', '') for s in servers}

        for name in found:
            if name not in existing_names:
                servers.append({
                    'name': name,
                    'transport': '',
                    'url': '',
                    'command': '',
                    'description': f'MCP server referenced in skill as mcp__{name}__*',
                })
            else:
                # Existing server without transport — don't fabricate
                for s in servers:
                    if s.get('name') == name and not s.get('transport'):
                        s['transport'] = ''
        return servers

    def _detect_api_templates(self, body: str) -> list[dict[str, Any]]:
        """Detect curl commands in SKILL.md body and extract API templates."""
        templates: list[dict[str, Any]] = []
        seen_urls: set[str] = set()

        for match in self._CURL_PATTERN.finditer(body):
            url = match.group(1)

            if url in seen_urls:
                continue
            seen_urls.add(url)

            context_start = max(0, match.start() - 200)
            context_end = min(len(body), match.end() + 200)
            context = body[context_start:context_end]

            method = "GET"
            if "-d " in context or "--data " in context or "--data-raw " in context:
                method = "POST"

            templates.append({
                "name": self._derive_api_name(url, method),
                "url": url,
                "context": context.strip(),
                "method": method,
            })

        return templates

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

        # Agent section stub
        agent_config = ahs_dict.get("agent", {})
        if isinstance(agent_config, dict) and agent_config:
            agent = AgentConfig(runtime=AgentRuntimeConfig(**agent_config))
        else:
            agent = None

        return AHSSpec(
            identity=identity,
            intent=intent,
            interface=interface,
            base=base,
            instructions=instructions,
            resources=Resources(**ahs_dict.get("resources", {})),
            composition=composition,
            agent=agent,
        )
