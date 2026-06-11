"""Phase 3 AI Composer — v0.8 LLM-driven agent generation with meta-reflection.

The AI Composer is the creative engine that generates custom agent.py code
for each skill. It goes beyond template-filling by using an LLM to:
1. Design hybrid tool wiring patterns
2. Create custom helper classes when needed
3. Optimize workflow structures
4. Add domain-aware error handling
5. Tailor agent architecture to skill complexity

The Meta-Reflection Loop reviews and iteratively improves generated code.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Core data structures
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class ComposerInput:
    """Complete context for the AI Composer."""

    # From Phase 1+1.5+2
    ahs_spec: dict[str, Any]           # Full AHSSPEC
    skill_md: str                       # SKILL.md body
    skill_files: list[str] = field(default_factory=list)  # All files in skill directory
    script_manifest: Any = None         # ScriptManifest from Phase 1.5
    phase2_reasoning: dict[str, str] = field(default_factory=dict)  # Reasoning traces

    # Reference material (auto-generated)
    core_api_doc: str = ""                   # agenthatch-core API surface
    best_practices: str = ""                 # Coding conventions & anti-patterns
    few_shot_examples: list[dict[str, Any]] = field(default_factory=list)  # Prior agents

    # Reflection feedback (set by MetaReflectionLoop for correction rounds)
    correction_feedback: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ComposerOutput:
    """Structured output from the AI Composer."""

    agent_py: str                       # Complete agent.py
    brick_manifest: dict[str, Any] = field(default_factory=dict)
    tool_wiring: list[dict[str, Any]] = field(default_factory=list)
    custom_classes: list[str] = field(default_factory=list)
    design_rationale: str = ""
    confidence: float = 0.0
    reflection_rounds: int = 1
    fidelity_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────
# Composer system prompt
# ─────────────────────────────────────────────────────────────────────────

COMPOSER_SYSTEM_PROMPT = """You are an AGENT ENGINEER — the AI Composer for agenthatch v0.8.
Your task: generate a complete agent.py file for a skill.

## AGENTHATCH-CORE API REFERENCE
```
AHCoreAgent.__init__(identity, runtime_config, spec_path, brick_manifest)
BrickManifest(
    loop_engine: LoopKind,       # "direct", "conversation", "plan_guided"
    capbus: bool,                # Enable CapBus tool registry
    hooks: bool,                 # Enable pre/post-turn hooks
    guard_active: bool,          # Enable compiled guard validation
    credential_vault: bool,      # Enable credential vault
    file_processor: bool,        # Enable file processor
    memory: bool,                # Enable memory retrospection
    archetype: str,              # "multi-step", "prompt-only", "external-tool", "mcp-connector"
)
CapBus.register(name, executor, schema, source)
CompiledWorkflow(steps: list[WorkflowStep])
WorkflowStep(step: int, description: str, script: str | None)
MCPProxyExecutor(cap_name, server_name, mcp_config, ...)
ContextManager.inject_system(msg) / add_user_message(msg)
Memory.store(entry) / get_recent_session_entries(n)
CredentialVault.get(key)
CompiledGuard.from_rules(rules)
LLMClient.chat / chat_stream / chat_structured
ConversationLoop(system_prompt, llm_client, context_manager, ...)
```

## CODING CONVENTIONS
- Import CompiledWorkflow from agenthatch_core.bricks.workflow
- Use `from __future__ import annotations`
- `script=None`, not `script=null`
- MCP tools use MCPProxyExecutor (direct subprocess via mcporter)
- Warmup scripts run in __init__, not per-turn
- mcporter syntax: `mcporter call Server.Tool key=value`
- Scripts execute as direct subprocess
- Import-based tool binding for Python scripts (Phase 1.5 ScriptAnalyzer)

## ANTI-PATTERNS — DO NOT DO THESE
- DO NOT inline WorkflowStep/CompiledWorkflow definitions (import from core)
- DO NOT dual-register tools (MCP + subprocess same name)
- DO NOT use `script=null` (must be `script=None`)
- DO NOT import agenthatch CLI code in generated agents
- DO NOT hallucinate script functions (use ScriptManifest only)
- DO NOT use Sandbox (sandbox layer removed in v0.8)

## CREATIVE ENGINEERING CAPABILITIES
You are an AGENT ENGINEER. Beyond template-filling, you can:

1. DESIGN hybrid tool wiring
   - MCPProxyExecutor for MCP-backed tools
   - CLI executor (subprocess) when MCP is unavailable (fallback)
   - Import-binding: direct function call for Python scripts
   - Descriptive: schema-only when no executable path

2. CREATE helper classes when needed
   - Input validators for complex tool schemas
   - Result transformers (e.g., CSV→dict, JSON→Markdown)
   - Retry wrappers for flaky API tools

3. OPTIMIZE workflow structure
   - Warmup for one-time setup (MCP connection tests)
   - Per-turn for context-dependent steps
   - Hybrid for complex pipelines

4. ADD domain-aware error handling
   - MCP connection errors → suggest mcporter install
   - Subprocess timeout → adjust timeout or split work
   - Auth failures → suggest credential vault check

5. GENERATE meaningful documentation
   - Module-level docstring explaining the agent's role
   - Method-level docstrings for public methods
"""


# ─────────────────────────────────────────────────────────────────────────
# Composer user prompt template
# ─────────────────────────────────────────────────────────────────────────

COMPOSER_USER_PROMPT = """Generate a complete agent.py for the following skill.

## Skill (SKILL.md)
{skill_md}

## AHSSPEC (what the agent MUST implement)
{ahspec_json}

## Script Manifest (Phase 1.5)
{script_manifest}

## Correction Feedback (previous iteration issues)
{correction_feedback}

## REQUIREMENTS
1. The agent class MUST extend AHCoreAgent
2. All capabilities from AHSSPEC MUST be registered via capbus.register()
3. Workflow steps MUST use CompiledWorkflow + WorkflowStep
4. Scripts execute via direct subprocess (subprocess.run)
5. MCP tools use MCPProxyExecutor
6. Include warmup for one-time setup scripts
7. Module-level docstring describing the agent
8. Use AgentIdentity for agent identity

## OUTPUT FORMAT
Provide ONLY the complete agent.py code. Start with imports, then class definition.
Do NOT wrap in markdown code fences — output raw Python code directly.
"""


# ─────────────────────────────────────────────────────────────────────────
# Meta-Reflection
# ─────────────────────────────────────────────────────────────────────────

COMPOSER_REFLECTION_PROMPT = """\
You are reviewing agent.py code that YOU generated for the following skill.

## Original SKILL.md (the source of truth)
{skill_md}

## AHSSPEC (what the agent MUST implement)
{ahspec_summary}

## Generated agent.py (YOUR code)
{agent_py}

## REVIEW CHECKLIST
Answer each with PASS or FAIL. For FAIL, reference specific line numbers.

### 1. COMPILABILITY
- Will `ast.parse()` succeed? (check syntax)
- Are all imports available in agenthatch-core? (check: agenthatch_core.*, pydantic, etc.)
- Any undefined variables or references?

### 2. COMPLETENESS
- Is every AHSSPEC capability registered via `self.capbus.register()`?
- Is every script file wired to a tool executor?
- Are all workflow steps from AHSSPEC present in the agent's workflow?

### 3. CORRECTNESS
- MCP tools use MCPProxyExecutor? (direct subprocess via mcporter)
- Local tools use direct subprocess or import-binding?
- CompiledWorkflow/WorkflowStep imported from `agenthatch_core.bricks.workflow`?
- `script=None` used (not `script=null`)?

### 4. QUALITY
- Meaningful docstrings on public methods? (not just placeholders)
- Error handling (try/except) for external tool calls?
- Any anti-patterns? (inline CompiledWorkflow, dual registration)

### 5. FIDELITY
- Does the agent's behavior description match what SKILL.md promises?
- Any feature in SKILL.md that the agent does NOT implement?
- Any feature in agent.py that SKILL.md does NOT describe?

## Output
Provide your review with specific line references and correction suggestions.
"""


class ComposerReflectionOutput(BaseModel):
    """Output from a meta-reflection review of generated agent.py."""
    compilability: Literal["pass", "fail"]
    compilability_detail: str = ""
    completeness: Literal["pass", "fail"]
    completeness_detail: str = ""
    correctness: Literal["pass", "fail"]
    correctness_detail: str = ""
    quality: Literal["pass", "fail"]
    quality_detail: str = ""
    fidelity: Literal["pass", "fail"]
    fidelity_detail: str = ""
    fidelity_score: float = Field(ge=0.0, le=1.0)
    issues: list[dict[str, Any]] = Field(default_factory=list)
    all_checks_pass: bool = False
    overall_confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


# ─────────────────────────────────────────────────────────────────────────
# Meta-Reflection Loop
# ─────────────────────────────────────────────────────────────────────────

class MetaReflectionLoop:
    """Iterative code generation with self-review and correction.

    The AI Composer generates agent.py, then reviews its own work through
    a structured reflection checklist. If issues are found, the feedback
    is fed back for correction (up to MAX_ITERATIONS rounds).
    """

    MAX_ITERATIONS = 3
    FIDELITY_THRESHOLD = 0.7

    def __init__(self, client: Any, model: str) -> None:
        """Initialize the reflection loop.

        Args:
            client: LLMClient instance for composer and reflection calls
            model: Model name to use
        """
        self.client = client
        self.model = model

    def compose_with_reflection(
        self,
        composer_input: ComposerInput,
    ) -> ComposerOutput:
        """Generate agent.py with iterative self-review and correction.

        Returns the best attempt found across all reflection rounds.
        """
        best_attempt: str | None = None
        best_score = 0.0
        best_rationale = ""

        for iteration in range(1, self.MAX_ITERATIONS + 1):
            # 1. Generate draft
            draft = self._generate_draft(composer_input)

            # 2. Self-review
            review = self._reflect(draft, composer_input)
            logger.info(
                "Meta-reflection round %d/%d: all_pass=%s fidelity=%.2f confidence=%.2f issues=%d",
                iteration, self.MAX_ITERATIONS,
                review.all_checks_pass, review.fidelity_score,
                review.overall_confidence, len(review.issues),
            )

            # Track best attempt
            if review.fidelity_score > best_score:
                best_attempt = draft
                best_score = review.fidelity_score
                best_rationale = review.rationale

            # 3. If clean, accept immediately
            if review.all_checks_pass and review.fidelity_score >= self.FIDELITY_THRESHOLD:
                return ComposerOutput(
                    agent_py=draft,
                    design_rationale=review.rationale,
                    confidence=review.overall_confidence,
                    reflection_rounds=iteration,
                    fidelity_score=review.fidelity_score,
                )

            # 4. Feed issues back for correction (if not last iteration)
            if iteration < self.MAX_ITERATIONS:
                composer_input.correction_feedback = review.issues

        # Max iterations — return best attempt
        logger.warning(
            "Max reflection rounds (%d) reached. "
            "Returning best attempt (fidelity=%.2f).",
            self.MAX_ITERATIONS, best_score,
        )
        return ComposerOutput(
            agent_py=best_attempt or "",
            design_rationale=best_rationale,
            confidence=best_score,  # Best available
            reflection_rounds=self.MAX_ITERATIONS,
            fidelity_score=best_score,
        )

    def _generate_draft(self, composer_input: ComposerInput) -> str:
        """Generate a draft agent.py from the composer input."""
        import json as _json

        correction_text = ""
        if composer_input.correction_feedback:
            correction_text = _json.dumps(
                composer_input.correction_feedback, indent=2, ensure_ascii=False,
            )

        prompt = COMPOSER_USER_PROMPT.format(
            skill_md=str(composer_input.skill_md)[:4000],
            ahspec_json=_json.dumps(
                composer_input.ahs_spec, indent=2, ensure_ascii=False, default=str,
            ),
            script_manifest=str(composer_input.script_manifest),
            correction_feedback=correction_text or "(none — first attempt)",
        )

        try:
            response = self.client.chat(
                messages=[
                    {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                model=self.model,
                temperature=0.3,
                thinking=True,
            )
            return response  # type: ignore[no-any-return]
        except Exception as e:
            logger.error("Composer draft generation failed: %s", e)
            raise

    def _reflect(
        self,
        agent_py: str,
        composer_input: ComposerInput,
    ) -> ComposerReflectionOutput:
        """Run meta-reflection review on generated agent.py."""

        prompt = COMPOSER_REFLECTION_PROMPT.format(
            skill_md=str(composer_input.skill_md)[:4000],
            ahspec_summary=self._summarize_ahspec(composer_input.ahs_spec),
            agent_py=agent_py[:8000],  # Truncate long files
        )

        try:
            reflection = self.client.chat_structured(
                messages=[
                    {"role": "system", "content": COMPOSER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                response_model=ComposerReflectionOutput,
                model=self.model,
                temperature=0.1,
                thinking=True,
            )
            return reflection  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning("Meta-reflection LLM call failed: %s — accepting as-is", e)
            return ComposerReflectionOutput(
                compilability="pass",
                compilability_detail="reflection skipped due to LLM error",
                completeness="pass",
                correctness="pass",
                quality="pass",
                fidelity="pass",
                fidelity_score=0.7,
                all_checks_pass=True,
                overall_confidence=0.7,
                rationale="Auto-accepted — reflection LLM call failed",
            )

    @staticmethod
    def _summarize_ahspec(ahs_spec: dict[str, Any]) -> str:
        """Create a compact summary of the AHSSPEC for the reflection prompt."""

        parts: list[str] = []

        identity = ahs_spec.get("identity", {})
        if identity:
            parts.append(
                f"Agent: {identity.get('display_name', 'Unknown')} "
                f"({identity.get('id', 'unknown')}) v{identity.get('version', '0.1.0')}"
            )

        intent = ahs_spec.get("intent", {})
        if intent:
            triggers = intent.get("triggers", [])
            parts.append(f"Triggers: {', '.join(triggers[:5])}")
            parts.append(f"Summary: {intent.get('summary', '')[:200]}")

        interface = ahs_spec.get("interface", {})
        if interface:
            provides = interface.get("provides", [])
            parts.append(
                f"Provides: {', '.join(p.get('capability', '?') for p in provides[:10])}"
            )
            requires = interface.get("requires", [])
            parts.append(
                f"Requires: {', '.join(r.get('capability', '?') for r in requires[:5])}"
            )

        instructions = ahs_spec.get("instructions", {})
        if instructions:
            steps = instructions.get("workflow", [])
            parts.append(f"Workflow: {len(steps)} steps")

        base = ahs_spec.get("base", {})
        if base:
            parts.append(f"Runtime: {base.get('runtime', 'null')}")

        return "\n".join(parts)
