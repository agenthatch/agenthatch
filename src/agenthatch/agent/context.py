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


def _extract_balanced_json(text: str) -> list[str]:
    r"""Extract balanced JSON objects from text using brace-depth counting.

    Unlike re.findall(r'\{[\s\S]*?\}', text), this correctly handles
    nested objects, arrays, and strings containing braces.
    """
    results: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\' and in_string:
            escape = True
            continue
        if ch == '"' and not escape:
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == '{':
            if depth == 0:
                start = i
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                results.append(text[start:i + 1])
                start = -1
        elif depth < 0:
            depth = 0

    return results


class ContextManager:
    """Constructs system prompt and manages conversation history window."""

    _CHARS_PER_TOKEN_ESTIMATE: int = 4

    COMPACT_RATIO: float = 0.75
    COMPACT_MIN_SAVINGS_RATIO: float = 0.30
    MIN_RECENT_TURNS: int = 3
    MAX_CONSECUTIVE_FAILURES: int = 3

    MAX_TOOL_RESULTS_FULL: int = 10
    TOOL_RESULT_SUMMARY_CHARS: int = 200

    ANCHOR_RULES: list[str] = []
    _skill_dir: Path | None = None

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
        self._turn_count: int = 0
        self._batch_total: int = 0
        self._batch_current: int = 0
        self._capbus: Any = None
        self._rich_prompt: bool = False

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

    def build_system_prompt(self, rich: bool = False) -> str:
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

        # ── DD-05-10: Operational guidance from raw_body ──
        if self.spec.instructions.raw_body:
            body = self.spec.instructions.raw_body
            if len(body) > 3000:
                body = body[:3000] + "\n\n... (truncated for context window)"
            parts.append("\n## Operational Guidance")
            parts.append(body)

        # ── DD-05-09: Resource summary ──
        if self.spec.resources.references:
            parts.append("\n## Reference Documents")
            parts.append(
                "These documents contain domain knowledge. "
                "Read them with the read_file tool when you need "
                "detailed information."
            )
            for ref in self.spec.resources.references:
                parts.append(
                    f"- {ref.get('name', ref.get('path', 'unknown'))}"
                )
        if self.spec.resources.scripts:
            parts.append("\n## Available Scripts")
            parts.append(
                "These scripts are available via run_skill_script(). "
                "Call them by script_name."
            )
            for script in self.spec.resources.scripts:
                parts.append(
                    f"- run_skill_script("
                    f'script_name="{script.get("name", "")}"'
                    f")"
                )

        # ── DD-05-21: External tool summary ──
        if self._capbus is not None:
            external_tools = []
            for name, cap in self._capbus.capabilities.items():
                if cap.type == "external":
                    desc = cap.schema.get("description", name)
                    external_tools.append(f"- {name}: {desc}")
            if external_tools:
                parts.append("\n## Available External Tools")
                parts.append(
                    "These tools connect to external services. "
                    "Use them to gather data, query APIs, and interact "
                    "with infrastructure."
                )
                parts.extend(external_tools)

        if self.spec.instructions.output_template:
            parts.append("")
            parts.append(
                "## Output Format (MANDATORY — enforced every turn)"
            )
            # ── DD-05-19: Template guard — reinforced every turn ──
            if self.summary is not None:
                parts.append(
                    "This is a long-running session. "
                    "You MUST still follow the output template below. "
                    "Do not drift into free-form text."
                )
            parts.append(
                "You MUST format your final answer EXACTLY according "
                "to the template below. Do not add, remove, or reorder "
                "fields. Replace {placeholders} with actual values."
            )
            parts.append(f"\n{self.spec.instructions.output_template}")

        # ── DD-05-18: Anchor rules survive compaction ──
        should_inject = self.summary is not None or (
            self._turn_count > 0 and self._turn_count % 20 == 0
        )
        if self.ANCHOR_RULES and should_inject:
            parts.append("\n## Core Rules (NEVER forget)")
            parts.append(
                "The following rules are your foundational constraints. "
                "They were established at the start of this session and "
                "MUST be followed regardless of context compaction."
            )
            for rule in self.ANCHOR_RULES:
                parts.append(f"- {rule}")

        # ── DD-05-20: Batch progress gating ──
        if self._batch_total > 0:
            parts.append("")
            parts.append("## Progress Tracking")
            parts.append(
                f"You are processing a batch of {self._batch_total} items. "
                f"Current progress: {self._batch_current}/{self._batch_total}. "
                "After completing each item, clearly state your progress "
                "and continue to the next item. Stop when all items are done. "
                "Do NOT repeat already-completed items."
            )

        # ── DD-05-38: Rich prompt mode — reference summaries ──
        if rich and self.spec.resources.references and self._skill_dir:
            parts.append("\n## Reference Summaries")
            for ref in self.spec.resources.references[:5]:
                ref_path = ref.get("name", "")
                ref_file = self._skill_dir / ref_path
                if ref_file.exists():
                    content = ref_file.read_text()[:500]
                    parts.append(f"\n### {ref_path}\n{content}")

        return "\n".join(parts)

    def build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """Build complete message list (system + history + user).

        v0.5: Tool context isolation — old tool results truncated to
        TOOL_RESULT_SUMMARY_CHARS to prevent prompt bloat.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": self.build_system_prompt(rich=self._rich_prompt)}
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

        # ── DD-05-04: Reorder orphaned tool messages instead of stripping ──
        reordered: list[dict[str, Any]] = []
        pending_orphan_tools: list[dict[str, Any]] = []
        last_assistant_had_tool_calls = False
        last_assistant_tc_msg: dict[str, Any] | None = None

        for msg in isolated:
            role = msg.get("role", "")
            if role == "tool":
                if last_assistant_had_tool_calls:
                    reordered.append(msg)
                else:
                    pending_orphan_tools.append(msg)
            elif role == "assistant":
                if msg.get("tool_calls"):
                    last_assistant_had_tool_calls = True
                    last_assistant_tc_msg = msg
                    reordered.append(msg)
                    if pending_orphan_tools:
                        reordered.extend(pending_orphan_tools)
                        pending_orphan_tools[:] = []
                else:
                    last_assistant_had_tool_calls = False
                    if pending_orphan_tools:
                        pending_orphan_tools[:] = []
                    reordered.append(msg)
            else:
                reordered.append(msg)

        if pending_orphan_tools and last_assistant_tc_msg:
            logger.warning(
                "Appending %d orphaned tool messages after last assistant+tool_calls",
                len(pending_orphan_tools),
            )
            idx = reordered.index(last_assistant_tc_msg) + 1
            reordered[idx:idx] = pending_orphan_tools

        isolated = reordered

        # ── DD-08-02 Part B: Validate tool call chain completeness ──
        validated: list[dict[str, Any]] = []
        pending_tc_ids: dict[str, list[int]] = {}
        tool_results: set[str] = set()

        for msg in isolated:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                validated.append(msg)
                for tc in msg["tool_calls"]:
                    tc_id = tc.get("id", "")
                    if tc_id:
                        pending_tc_ids.setdefault(tc_id, []).append(len(validated) - 1)
            elif msg.get("role") == "tool":
                tc_id = msg.get("tool_call_id", "")
                if tc_id and tc_id in pending_tc_ids:
                    tool_results.add(tc_id)
                    validated.append(msg)
            else:
                validated.append(msg)

        # Strip tool_calls without matching tool results
        for msg in validated:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                filtered_tcs = [
                    tc for tc in msg["tool_calls"]
                    if tc.get("id", "") in tool_results
                ]
                if not filtered_tcs:
                    msg.pop("tool_calls", None)
                elif len(filtered_tcs) != len(msg["tool_calls"]):
                    msg["tool_calls"] = filtered_tcs

        isolated = validated

        messages.extend(isolated)

        # Normalize: assistant messages with tool_calls must have content=None
        for msg in messages:
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                if msg.get("content") == "":
                    msg["content"] = None

        messages.append({"role": "user", "content": user_input})
        return messages

    def add_to_history(
        self,
        role: str,
        content: str | None,
        tool_call_id: str | None = None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
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
        """Truncate history to recent turns, preserving tool call message chains."""
        keep_count = self.MIN_RECENT_TURNS * 2

        if len(self.history) <= keep_count:
            return False

        # Walk backwards to find a safe truncation boundary.
        safe_boundary = len(self.history) - keep_count

        # Extend backwards to include the start of any tool call chain
        # that spans the boundary: assistant(tool_calls) → tool → tool → ... → assistant
        for i in range(safe_boundary - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                safe_boundary = i
                break
            elif msg.get("role") == "tool":
                continue
            else:
                break

        # Ensure tool results for tool_calls in kept section are included
        tool_call_ids: set[str] = set()
        for i in range(safe_boundary, len(self.history)):
            msg = self.history[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    tool_call_ids.add(tc.get("id", ""))
            elif msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid in tool_call_ids:
                    tool_call_ids.discard(tid)

        if tool_call_ids:
            for i in range(safe_boundary - 1, -1, -1):
                msg = self.history[i]
                if msg.get("role") == "assistant" and msg.get("tool_calls"):
                    safe_boundary = i
                    break

        self.history = self.history[safe_boundary:]
        logger.debug(
            "Truncated history to %d messages (preserved tool call chains)",
            len(self.history),
        )
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

        # ── DD-05-18: Inject anchor rules into compact prompt ──
        if self.ANCHOR_RULES:
            prompt += (
                "\n\nIMPORTANT: Preserve these rules in your summary: "
                + "; ".join(self.ANCHOR_RULES)
            )

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
            # ── DD-05-40: Brace-balanced extraction (replaces non-greedy regex) ──
            blocks = _extract_balanced_json(text)
            if blocks:
                blocks.sort(key=lambda b: -len(b))
                for block in blocks[:3]:
                    try:
                        data = json.loads(block)
                        if isinstance(data, dict) and "session_intent" in data:
                            return CompactSummary(**data)
                    except (json.JSONDecodeError, TypeError):
                        continue
            # Fall back to regex if brace-balancing found nothing
            regex_blocks = re.findall(r'\{[\s\S]*?\}', text)
            for block in regex_blocks[:3]:
                try:
                    data = json.loads(block)
                    if isinstance(data, dict) and "session_intent" in data:
                        return CompactSummary(**data)
                except (json.JSONDecodeError, TypeError):
                    continue
            raise json.JSONDecodeError("No valid JSON found in response", text, 0)

        raise RuntimeError("No LLM client available for compaction")

    def set_batch_scope(self, total: int) -> None:
        self._batch_total = total
        self._batch_current = 0

    def advance_batch(self) -> None:
        self._batch_current += 1
