"""AHS v1.1 Specification — Pydantic data models.

The complete AHSSPEC schema drives both v0.3 standardization output
and v0.4 SkillAgent initialization (via SkillAgent.from_ahspec()).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# ─── Phase 1 Data Structures (non-middleware) ──────────────────────────

@dataclass
class FileEntry:
    """A single file discovered in the skill directory.

    Deliberately does NOT include extension classification.
    Semantic classification is LLM's responsibility (Phase 2 Harness).
    """
    path: str                       # relative path (e.g. "tools/update.sh")
    hash: str                       # SHA-256 hex digest
    size_bytes: int
    content: str | None = None      # text content (None = binary/unreadable)


@dataclass
class FileManifest:
    """Flat list of all files in a skill directory.

    Replaces the old 4-category model (scripts/references/assets/other).
    No classification — that's LLM's job.
    """
    entries: list[FileEntry] = field(default_factory=list)
    entrypoint: str = ""            # relative path of SKILL.md (or variant)

    def content_bundle(self) -> list[dict[str, str | None]]:
        """All readable file contents, ready for Harness consumption."""
        return [
            {"path": e.path, "content": e.content}
            for e in self.entries
            if e.content is not None
        ]


@dataclass
class ContextPack:
    """Phase 1 output — ephemeral, zero semantic transformation."""
    frontmatter: dict[str, Any] | None
    body: str
    file_manifest: FileManifest
    dir_name: str
    parse_warnings: list[str] = field(default_factory=list)
    skill_dir: Path | None = None  # v0.8: skill directory path for Phase 1.5 ScriptAnalyzer


# ─── Harness Contracts ─────────────────────────────────────────────────

# v0.9.17: Single source of truth for harness key → human-readable label.
# Consumed by hatch.py renderers, validate.py repair router, and report.py.
HARNESS_LABELS: dict[str, str] = {
    "A": "extract_identity",
    "B": "infer_intent",
    "C": "infer_interface",
    "D": "detect_base_and_instructions",
    "E": "assemble_and_validate",
    "F": "infer_mcp_servers",
}


@dataclass
class HarnessOutput:
    """Unified output contract for all 6 AgentHarnesses.

    v0.9.17: token_usage captures LLM token consumption for the hatch report.
    Populated by AgentHarness.run() from client.last_usage after each LLM call.
    Empty dict when no LLM call was made (e.g. Harness F regex fallback).
    """
    result: dict[str, Any]
    confidence: float
    reasoning_trace: list[str]
    self_check_passed: bool
    degradation_applied: list[str] = field(default_factory=list)
    internal_retries: int = 0
    token_usage: dict[str, int] = field(default_factory=dict)


# ─── AHSSPEC v1.1 Schema ───────────────────────────────────────────────

class Identity(BaseModel):
    """Skill identity."""
    id: str            # kebab-case, globally unique
    display_name: str
    version: str = ""  # v0.8.2: deprecated — use agent.hatched_at instead

    @field_validator("id")
    @classmethod
    def validate_kebab_case(cls, v: str) -> str:
        if not re.match(r"^[a-z0-9]+(-[a-z0-9]+)*$", v):
            raise ValueError(f"id '{v}' is not kebab-case")
        return v

    @field_validator("display_name")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("display_name must not be empty")
        return v.strip()


class Intent(BaseModel):
    """Skill intent — triggers & satisfies."""
    triggers: list[str]           # 5-15 keywords
    satisfies: list[str]          # 3-8 intent templates with {param}
    summary: str


CAPABILITY_TYPES = Literal[
    "data", "analysis", "media", "transform",
    "action", "event", "knowledge", "renderer",
]


class Capability(BaseModel):
    """A single capability entry in provides/requires."""
    capability: str    # snake_case, globally unique
    type: CAPABILITY_TYPES
    input_schema: dict[str, Any] = {}
    output_schema: dict[str, Any] = {}   # v0.7.6: JSON Schema for tool output validation


class MCPServerEntry(BaseModel):
    """Single MCP server reference from Harness F inference."""
    name: str
    transport: str = ""          # "stdio", "streamable_http", "sse", or ""
    url: str = ""
    command: str = ""
    description: str = ""


class InferMCPServersOutput(BaseModel):
    """Harness F structured output."""
    mcp_servers: list[MCPServerEntry] = Field(default_factory=list)


class APIParam(BaseModel):
    """API template parameter."""
    name: str
    type: str
    required: bool = True


class APITemplate(BaseModel):
    """API template auto-detected from curl commands."""
    name: str
    url: str
    method: str = "GET"
    params: list[APIParam] = []
    headers: dict[str, str] = {}
    auth_env_var: str | None = None


class Interface(BaseModel):
    """Skill interface — provides, requires, compatible_with, MCP, API templates."""
    provides: list[Capability]
    requires: list[Capability] = []
    compatible_with: list[str] = []
    mcp_servers: list[MCPServerEntry] = []
    api_templates: list[APITemplate] = []


class EnvVar(BaseModel):
    """Environment variable definition."""
    name: str
    required: bool = False
    description: str = ""


class WorkflowStep(BaseModel):
    """A single step in the skill's workflow."""
    step: int
    description: str
    script: str | None = None


class Safety(BaseModel):
    """Safety/guardrail configuration."""
    confirmation_required_for: list[str] = []
    plan_required: bool = False
    max_rows_default: int | None = None
    parameterized_only: bool = False


class Instructions(BaseModel):
    """Skill instructions — workflow, rules, safety, output template."""
    workflow: list[WorkflowStep] = []
    rules: list[str] = []
    safety: Safety = Safety()
    output_template: str | None = None
    raw_body: str | None = None


class BaseSpec(BaseModel):
    """Runtime base specification."""
    runtime: str | None = None     # python3.11, bash, node20, or null (pure instruction)
    timeout: str = "60s"
    env: list[EnvVar] = []
    dependencies: list[str] = []

    @field_validator("timeout", mode="before")
    @classmethod
    def _coerce_timeout(cls, v: Any) -> str:
        """Coerce int/float timeout values to string format."""
        if isinstance(v, (int, float)):
            return f"{int(v)}s"
        if isinstance(v, str):
            cleaned = v.strip()
            if not cleaned:
                return "60s"
            normalized = cleaned.rstrip("s") + "s"
            try:
                int(normalized.rstrip("s"))
                return normalized
            except ValueError:
                return "60s"
        return "60s"


def _coerce_base_data(base_data: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw dict values to BaseSpec-compatible types.

    Harness E uses unstructured chat() — LLM may produce type mismatches
    (int timeout, str runtime, single string dependencies, dict env, etc.)
    that Pydantic strict validation rejects. This normalizes all known
    type mismatches before BaseSpec(**data).
    """
    data = base_data.copy() if base_data else {}

    # timeout: int 60 → "60s", str "45" → "45s", None/{}/[] → "60s"
    if "timeout" in data:
        val = data["timeout"]
        if isinstance(val, (int, float)):
            data["timeout"] = f"{int(val)}s"
        elif isinstance(val, str):
            cleaned = val.strip()
            if cleaned:
                normalized = cleaned.rstrip("s") + "s"
                try:
                    int(normalized.rstrip("s"))
                    data["timeout"] = normalized
                except ValueError:
                    data["timeout"] = "60s"
            else:
                data["timeout"] = "60s"
        elif isinstance(val, (dict, list)):
            data["timeout"] = "60s"
        elif val is None:
            data["timeout"] = "60s"

    # runtime: normalize, reject invalid values
    VALID_RUNTIMES = {"python3.11", "bash", "node20"}
    if "runtime" in data and isinstance(data["runtime"], str):
        cleaned = data["runtime"].lower().strip()
        if cleaned in ("python", "python3"):
            cleaned = "python3.11"
        if cleaned not in VALID_RUNTIMES:
            data["runtime"] = None
        else:
            data["runtime"] = cleaned

    # dependencies: "numpy" → ["numpy"]
    if "dependencies" in data and isinstance(data["dependencies"], str):
        deps = [d.strip() for d in data["dependencies"].split(",") if d.strip()]
        data["dependencies"] = deps if deps else []

    # env: {"KEY": "val"} → [{"name": "KEY", "description": "val"}]
    if "env" in data and isinstance(data["env"], dict):
        if "name" in data["env"] or "description" in data["env"]:
            # {"description": "..."} or {"name": "X", "description": "Y"}
            if "name" not in data["env"]:
                data["env"]["name"] = ""
            data["env"] = [data["env"]]
        else:
            # {"KEY": "val"}
            data["env"] = [
                {"name": k, "description": str(v)}
                for k, v in data["env"].items()
            ]

    # env: list items missing "name" → fill with ""
    if "env" in data and isinstance(data["env"], list):
        for item in data["env"]:
            if isinstance(item, dict) and "name" not in item:
                item["name"] = ""

    return data


def _coerce_ahs_dict(spec_dict: dict[str, Any]) -> dict[str, Any]:
    """Coerce type mismatches across all AHSSPEC fields.

    Extends _coerce_base_data to cover identity, intent, and interface.
    """
    data = spec_dict.copy() if spec_dict else {}

    # ── identity coercion ──
    identity = data.get("identity", {})
    if isinstance(identity, dict):
        if "id" in identity and isinstance(identity["id"], str):
            identity["id"] = re.sub(r"[^a-z0-9-]", "-", identity["id"].lower()).strip("-")
        if "version" in identity and isinstance(identity["version"], str):
            # "v1.2.3" → "1.2.3"
            v = identity["version"].lstrip("vV")
            if re.match(r"^\d+\.\d+\.\d+$", v):
                identity["version"] = v

    # ── intent coercion ──
    intent = data.get("intent", {})
    if isinstance(intent, dict):
        # triggers: comma-separated string → list
        if "triggers" in intent and isinstance(intent["triggers"], str):
            intent["triggers"] = [t.strip() for t in intent["triggers"].split(",") if t.strip()]
        # satisfies: comma-separated string → list
        if "satisfies" in intent and isinstance(intent["satisfies"], str):
            intent["satisfies"] = [s.strip() for s in intent["satisfies"].split(",") if s.strip()]

    # ── interface coercion ──
    interface = data.get("interface", {})
    if isinstance(interface, dict):
        # provides: single dict → list
        if "provides" in interface and isinstance(interface["provides"], dict):
            interface["provides"] = [interface["provides"]]
        # requires: single dict → list
        if "requires" in interface and isinstance(interface["requires"], dict):
            interface["requires"] = [interface["requires"]]
        # compatible_with: comma-separated string → list
        if "compatible_with" in interface and isinstance(interface["compatible_with"], str):
            interface["compatible_with"] = [
                s.strip() for s in interface["compatible_with"].split(",") if s.strip()
            ]

    # ── base coercion (existing) ──
    if "base" in data:
        data["base"] = _coerce_base_data(data["base"])

    return data


class Modes(BaseModel):
    """Multi-mode skill configuration."""
    modes: dict[str, dict[str, Any]] = {}


class Resources(BaseModel):
    """Skill resources (generated from file_manifest + SHA-256)."""
    scripts: list[dict[str, str]] = []
    references: list[dict[str, str]] = []
    assets: list[dict[str, str]] = []


class EventListener(BaseModel):
    """Event listener for composition layer."""
    event: str
    from_: str = ""
    action: str = ""


class Composition(BaseModel):
    """Composition layer — event listeners for cross-skill orchestration."""
    event_listeners: list[EventListener] = []


class ConfidenceReport(BaseModel):
    """Overall confidence breakdown."""
    overall: float
    per_harness: dict[str, float] = {}


# ─── v0.4 Agent Runtime Config ──────────────────────────────────────────

class CompactConfig(BaseModel):
    """Per-skill auto-compact configuration (v0.5)."""
    enabled: bool = True
    ratio: float = 0.75
    min_recent_turns: int = 3
    min_savings_ratio: float = 0.30


class AgentRuntimeConfig(BaseModel):
    """Skill-level Agent runtime configuration (added in v0.4)."""
    provider: str | None = None
    model: str | None = None
    env: dict[str, str] = {}
    temperature: float = 0.7
    max_tokens: int = 4096
    features: dict[str, bool] = {}
    compact: CompactConfig | None = None  # v0.5 NEW


class AgentConfig(BaseModel):
    """Agent lifecycle metadata (v0.8.2: status redesigned)."""
    status: Literal["unhatched", "hatched"] = "unhatched"
    hatched_at: datetime | None = None     # set when Phase 2 completes
    generated_at: datetime | None = None   # set when Phase 3 completes


class AHSSpec(BaseModel):
    """Complete AHSSPEC v1.1 — the single middleware artifact.

    This is what v0.3 outputs and v0.4 SkillAgent.from_ahspec() consumes.
    """
    identity: Identity
    intent: Intent
    interface: Interface
    base: BaseSpec = BaseSpec()
    instructions: Instructions = Instructions()
    resources: Resources = Resources()
    modes: Modes | None = None
    composition: Composition = Composition()
    agent: AgentConfig | None = None   # added in v0.4

    confidence_report: ConfidenceReport | None = None
    harness_traces: list[HarnessOutput] = []


# ─── Pydantic Models for Harness Structured Output ─────────────────────

class IdentityOutput(BaseModel):
    """Harness A structured output."""
    identity: Identity


class IntentOutput(BaseModel):
    """Harness B structured output."""
    intent: Intent


class InterfaceOutput(BaseModel):
    """Harness C structured output."""
    interface: Interface


class BaseAndInstructionsOutput(BaseModel):
    """Harness D structured output."""
    base: BaseSpec
    instructions: Instructions


class AssembleOutput(BaseModel):
    """Harness E structured output."""
    ahs_spec: dict[str, Any]
    confidence_report: ConfidenceReport | None = None
    warnings: list[str] = Field(default_factory=list)
