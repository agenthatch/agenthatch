"""ContextManager — System Prompt construction and conversation history (v0.4)."""

from __future__ import annotations

from typing import Any

from agenthatch.skill.spec import AHSSpec


class ContextManager:
    """Constructs system prompt and manages conversation history window."""

    _CHARS_PER_TOKEN_ESTIMATE: int = 4

    def __init__(self, ahs_spec: AHSSpec):
        self.spec = ahs_spec
        self.history: list[dict[str, Any]] = []
        self.max_history_turns = 20

    def build_system_prompt(self) -> str:
        """Build system prompt from AHSSPEC with domain persona injection."""
        parts: list[str] = []

        provides = [c.capability for c in self.spec.interface.provides]
        triggers = self.spec.intent.triggers

        parts.append(
            f"You are {self.spec.identity.display_name}, "
            f"a specialist agent created by agenthatch."
        )
        if self.spec.identity.author:
            parts.append(f"Author: {self.spec.identity.author}")

        parts.append("")
        parts.append("## Your Core Identity")
        if self.spec.intent.summary:
            parts.append(f"{self.spec.intent.summary}")
        parts.append("")
        parts.append(
            "You are NOT a general-purpose assistant. "
            "You specialize in your domain only."
        )
        if triggers:
            parts.append(
                f"You respond to: {', '.join(triggers[:8])}"
                f"{'...' if len(triggers) > 8 else ''}"
            )
        if provides:
            parts.append(
                f"You have these capabilities: {', '.join(provides)}."
            )
        parts.append(
            "If asked something clearly outside your domain, "
            "politely decline and suggest what you CAN help with."
        )

        if self.spec.instructions.workflow:
            parts.append("\n## Workflow")
            parts.append(
                "Follow these steps in order. Steps marked with "
                "-> Use tool require calling the specified tool."
            )
            for step in self.spec.instructions.workflow:
                line = f"{step.step}. {step.description}"
                if step.script:
                    line += (
                        f"\n   -> Use tool: run_skill_script("
                        f'script_name="{step.script}", ...)'
                    )
                parts.append(line)

        if self.spec.instructions.rules:
            parts.append("\n## Rules - NEVER violate these")
            for rule in self.spec.instructions.rules:
                parts.append(f"- {rule}")

        if self.spec.instructions.safety:
            s = self.spec.instructions.safety
            if s.plan_required:
                parts.append("\nAlways create a plan before executing.")

        if self.spec.instructions.output_template:
            parts.append(
                "\n## Output Format (MANDATORY - do not deviate)"
            )
            parts.append(
                "You MUST format your final answer EXACTLY according "
                "to the template below. Do not add, remove, or reorder "
                "fields. Replace {placeholders} with actual values."
            )
            parts.append(f"\n{self.spec.instructions.output_template}")

        return "\n".join(parts)

    def build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """Build complete message list (system + history + user)."""
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        recent = self.history[-self.max_history_turns * 2:]
        messages.extend(recent)
        messages.append({"role": "user", "content": user_input})
        return messages

    def add_to_history(self, role: str, content: str) -> None:
        """Add a message to conversation history."""
        self.history.append({"role": role, "content": content})

    def compress(self) -> str:
        """Context compression stub for v0.5."""
        return "(context_compressor not yet available)"

    def estimate_input_tokens(self) -> int:
        """Estimate total input tokens for current context.

        Uses 4 chars/token heuristic. Not exact, but sufficient
        for dynamic max_tokens adjustment (< 20% error margin).
        """
        system_len = len(self.build_system_prompt())
        history_len = sum(
            len(str(msg.get("content", "") or ""))
            for msg in self.history
        )
        return (system_len + history_len) // self._CHARS_PER_TOKEN_ESTIMATE
