"""ContextManager — System Prompt construction and conversation history (v0.5).

v0.5 additions:
- auto_compact_check() + compact() — LLM summarization with fallback truncation
- Tool context isolation — caps old tool results to prevent prompt bloat
- Summary merge — only one summary block after re-compaction
- Circuit breaker — 3 consecutive failures → skip compaction
- Per-skill compact config via agenthatch.yaml
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agenthatch.agent.compact import COMPACT_SYSTEM_PROMPT, CompactSummary
from agenthatch.agent.hooks import HookPoint
from agenthatch.skill.spec import AHSSpec

logger = logging.getLogger(__name__)


class ContextManager:
    """Constructs system prompt and manages conversation history window."""

    _CHARS_PER_TOKEN_ESTIMATE: int = 4

    COMPACT_RATIO: float = 0.75
    COMPACT_MIN_SAVINGS_RATIO: float = 0.30
    MIN_RECENT_TURNS: int = 3
    MAX_CONSECUTIVE_FAILURES: int = 3

    MAX_TOOL_RESULTS_FULL: int = 10
    TOOL_RESULT_SUMMARY_CHARS: int = 200

    def __init__(self, ahs_spec: AHSSpec):
        self.spec = ahs_spec
        self.history: list[dict[str, Any]] = []
        self.max_history_turns = 20

        self._consecutive_compact_failures: int = 0
        self.compact_config: dict[str, Any] | None = None
        self._hooks: Any = None
        self._state_manager: Any = None
        self._llm: Any = None
        self.summary: CompactSummary | None = None

    def _apply_compact_config(self) -> None:
        if self.compact_config:
            self.COMPACT_RATIO = self.compact_config.get(
                "ratio", self.COMPACT_RATIO
            )
            self.MIN_RECENT_TURNS = self.compact_config.get(
                "min_recent_turns", self.MIN_RECENT_TURNS
            )
            self.COMPACT_MIN_SAVINGS_RATIO = self.compact_config.get(
                "min_savings_ratio", self.COMPACT_MIN_SAVINGS_RATIO
            )

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
        """Build complete message list (system + history + user).

        v0.5: Tool context isolation — old tool results truncated to
        TOOL_RESULT_SUMMARY_CHARS to prevent prompt bloat.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt()}
        ]
        recent = self.history[-self.max_history_turns * 2:]

        tool_count = 0
        isolated: list[dict[str, Any]] = []
        for msg in reversed(recent):
            if msg.get("role") == "tool":
                tool_count += 1
                if tool_count > self.MAX_TOOL_RESULTS_FULL:
                    content = msg.get("content", "")
                    if isinstance(content, str) and len(content) > self.TOOL_RESULT_SUMMARY_CHARS:
                        msg = {
                            "role": "tool",
                            "tool_call_id": msg.get("tool_call_id", ""),
                            "content": content[:self.TOOL_RESULT_SUMMARY_CHARS] + "...",
                        }
            isolated.insert(0, msg)

        messages.extend(isolated)

        # Normalize: assistant messages with tool_calls must have content=None
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if not msg.get("content"):
                    msg["content"] = None

        messages.append({"role": "user", "content": user_input})
        return messages

    def add_to_history(self, role: str, content: str, tool_call_id: str | None = None, tool_calls: list[dict[str, Any]] | None = None) -> None:
        """Add a message to conversation history."""
        msg: dict[str, Any] = {"role": role, "content": content}
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        if tool_calls:
            msg["tool_calls"] = tool_calls
        self.history.append(msg)

    def compress(self) -> str:
        """DEPRECATED: context compression stub. Use compact() instead."""
        logger.warning("compress() is deprecated, use compact()")
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

    # ── v0.5 Auto-Compact ──────────────────────────────────────────────

    def auto_compact_check(self, max_tokens: int) -> bool:
        """Check if compaction should be triggered and execute it.

        Called ONCE per turn at ConversationLoop entry.
        Returns True if compaction was performed.
        """
        if self._consecutive_compact_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Circuit breaker open: %d consecutive compact failures. Skipping.",
                self._consecutive_compact_failures
            )
            return False

        current = self.estimate_input_tokens()
        if not self._should_compact(max_tokens, current):
            return False

        hook_ctx: dict[str, Any] = {"operation": "compact", "current_tokens": current}
        if self._hooks:
            self._hooks.execute(HookPoint.PRE_COMPACT, hook_ctx)
        if hook_ctx.get("skip"):
            return False

        result = self.compact(max_tokens)

        hook_ctx["success"] = result
        if self._hooks:
            self._hooks.execute(HookPoint.POST_COMPACT, hook_ctx)

        return result

    def _should_compact(self, max_tokens: int, current_tokens: int) -> bool:
        """Check if compaction is worth the token cost."""
        threshold = int(max_tokens * self.COMPACT_RATIO)
        if current_tokens < threshold:
            return False

        recent_tokens = sum(
            len(str(m.get("content", ""))) // 4
            for m in self.history[-(self.MIN_RECENT_TURNS * 2):]
        )
        estimated_after = 500 + recent_tokens
        savings_ratio = 1.0 - (estimated_after / current_tokens)

        if savings_ratio < self.COMPACT_MIN_SAVINGS_RATIO:
            logger.info(
                "Skipping compaction: estimated savings %.1f%% < %.0f%% minimum",
                savings_ratio * 100, self.COMPACT_MIN_SAVINGS_RATIO * 100,
            )
            return False

        return True

    def compact(self, max_tokens: int) -> bool:
        """Execute compaction. Returns True on success, False on fallback."""
        try:
            summary = self._generate_summary()
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Compact summary invalid (%s), falling back to truncation", e)
            self._consecutive_compact_failures += 1
            return self._fallback_truncation()

        required = ["session_intent", "current_state", "conversation_turns"]
        if not all(getattr(summary, f, None) for f in required):
            logger.warning("Compact summary missing required fields, truncating")
            self._consecutive_compact_failures += 1
            return self._fallback_truncation()

        summary.compressed_at = datetime.now().isoformat()
        self.summary = summary
        self._consecutive_compact_failures = 0

        offload_path = self._offload_full_history()

        recent = self.history[-(self.MIN_RECENT_TURNS * 2):]
        self.history = []
        self.history.append({
            "role": "system",
            "content": f"## Compaction Summary\n{self.summary.to_text()}"
        })
        self.history.extend(recent)

        logger.info(
            "Compaction complete: %d turns → summary (%d chars) + %d recent turns. "
            "Full history offloaded to %s",
            summary.conversation_turns, len(summary.to_text()),
            self.MIN_RECENT_TURNS, offload_path,
        )
        return True

    def _fallback_truncation(self) -> bool:
        """Simple truncation: keep only recent turns."""
        self.history = self.history[-(self.MIN_RECENT_TURNS * 2):]
        return False

    def _offload_full_history(self) -> Path | None:
        """Offload full conversation history to file system before truncation."""
        if self._state_manager is None:
            return None
        return self._state_manager.save_history(list(self.history))  # type: ignore[no-any-return]

    def _build_compact_messages(self) -> list[dict[str, Any]]:
        """Build messages for the compaction LLM call."""
        prompt = COMPACT_SYSTEM_PROMPT

        if self.summary:
            prompt += "\n\n## Prior Compaction Summary\n"
            prompt += self.summary.to_text()
            prompt += "\n\nUpdate this summary to include the recent conversation below."

        prompt += "\n\n## Conversation to Compact\n"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
        ]
        for msg in self.history:
            # Only include user and assistant messages for compaction.
            # Strip tool_calls from assistant messages (tool results are
            # summarized by the LLM into tool_calls_summary).
            role = msg.get("role", "")
            if role == "user":
                messages.append(msg)
            elif role == "assistant" and not msg.get("tool_calls"):
                messages.append(msg)
        return messages

    def _generate_summary(self) -> CompactSummary:
        """Call LLM to generate a CompactSummary from conversation history."""
        messages = self._build_compact_messages()

        try:
            if self._llm is not None:
                result = self._llm.chat_structured(
                    messages=messages,
                    response_model=CompactSummary,
                    max_retries=2,
                )
                return result  # type: ignore[no-any-return]
        except Exception:
            pass

        if self._llm is not None:
            text = self._llm.chat(
                messages=messages,
                temperature=0.1,
                max_tokens=1000,
            )
            match = re.search(r'\{[\s\S]*\}', text)
            if not match:
                raise json.JSONDecodeError("No JSON found in response", text, 0)
            data = json.loads(match.group(0))
            return CompactSummary(**data)

        raise RuntimeError("No LLM client available for compaction")
