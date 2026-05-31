"""AHS v1.1 Specification — Pydantic data models.

The complete AHSSPEC schema drives both v0.3 standardization output
and v0.4 SkillAgent initialization (via SkillAgent.from_ahspec()).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

# ─── Phase 1 Data Structures (non-middleware) ──────────────────────────

@dataclass
class FileEntry:
    """A single file discovered in the skill directory.

    Deliberately does NOT include extension classification.
    Semantic classification is LLM's responsibility (Phase 2 Harness).
    See DD-E01 for rationale.
    """
    path: str                       # relative path (e.g. "tools/update.sh")
    hash: str                       # SHA-256 hex digest
    size_bytes: int
    content: str | None = None      # text content (None = binary/unreadable)


@dataclass
class FileManifest:
    """Flat list of all files in a skill directory.

    Replaces the old 4-category model (scripts/references/assets/other).
    No classification — that's LLM's job (DD-E01).
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


# ─── Harness Contracts ─────────────────────────────────────────────────

@dataclass
class HarnessOutput:
    """Unified output contract for all 5 AgentHarnesses."""
    result: dict[str, Any]
    confidence: float
    reasoning_trace: list[str]
    self_check_passed: bool
    degradation_applied: list[str] = field(default_factory=list)
    internal_retries: int = 0


# ─── AHSSPEC v1.1 Schema ───────────────────────────────────────────────

class Identity(BaseModel):
    """Skill identity."""
    id: str            # kebab-case, globally unique
    display_name: str
    version: str       # semver
    license: str | None = None
    author: str | None = None
    meta: dict[str, Any] = {}


class Intent(BaseModel):
    """Skill intent — triggers & satisfies."""
    triggers: list[str]           # 5-15 keywords
    satisfies: list[str]          # 3-8 intent templates with {param}
    summary: str


class Capability(BaseModel):
    """A single capability entry in provides/requires."""
    capability: str    # snake_case, globally unique
    type: str          # data, analysis, media, transform, action, event, knowledge, renderer
    input_schema: dict[str, Any] = {}


class Interface(BaseModel):
    """Skill interface — provides, requires, compatible_with."""
    provides: list[Capability]
    requires: list[Capability] = []
    compatible_with: list[str] = []


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
    """Skill instructions — workflow, rules, safety."""
    workflow: list[WorkflowStep] = []
    rules: list[str] = []
    safety: Safety = Safety()
    output_template: str | None = None


class BaseSpec(BaseModel):
    """Runtime base specification."""
    runtime: str | None = None     # python3.11, bash, node20, or null (pure instruction)
    sandbox: bool = False
    timeout: str = "60s"
    env: list[EnvVar] = []
    dependencies: list[str] = []


def _coerce_base_data(base_data: dict[str, Any]) -> dict[str, Any]:
    """Coerce raw dict values to BaseSpec-compatible types.

    Harness E uses unstructured chat() — LLM may produce type mismatches
    (int timeout, str runtime, single string dependencies, dict env, etc.)
    that Pydantic strict validation rejects. This normalizes all known
    type mismatches before BaseSpec(**data).
    """
    data = base_data.copy() if base_data else {}

    # timeout: int 60 → "60s", str "45" → "45s", "60s " → "60s"
    if "timeout" in data:
        val = data["timeout"]
        if isinstance(val, (int, float)):
            data["timeout"] = f"{int(val)}s"
        elif isinstance(val, str):
            cleaned = val.strip().rstrip("s") + "s"
            try:
                int(cleaned.rstrip("s"))
                data["timeout"] = cleaned
            except ValueError:
                data["timeout"] = "60s"

    # sandbox: "true"/"false" → True/False
    if "sandbox" in data and isinstance(data["sandbox"], str):
        data["sandbox"] = data["sandbox"].lower().strip() in ("true", "yes", "1")

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
        if "name" not in data["env"] and "description" not in data["env"]:
            data["env"] = [
                {"name": k, "description": str(v)}
                for k, v in data["env"].items()
            ]

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
    """Skill 级别的 Agent 运行时配置 (v0.4 新增)."""
    provider: str | None = None
    model: str | None = None
    env: dict[str, str] = {}
    temperature: float = 0.7
    max_tokens: int = 4096
    features: dict[str, bool] = {}
    compact: CompactConfig | None = None  # v0.5 NEW


class AgentConfig(BaseModel):
    """Agent 段 (v0.4 新增)."""
    runtime: AgentRuntimeConfig = AgentRuntimeConfig()


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
    agent: AgentConfig | None = None   # v0.4 新增

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
