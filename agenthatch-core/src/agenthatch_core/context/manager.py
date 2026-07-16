"""ContextManager — System Prompt construction and conversation history (agenthatch-core)."""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from agenthatch_core.context.compact import COMPACT_SYSTEM_PROMPT, CompactSummary
from agenthatch_core.hooks import HookPoint

logger = logging.getLogger(__name__)


def _extract_balanced_json(text: str) -> list[str]:
    r"""Extract balanced JSON objects from text using brace-depth counting."""
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


CAPBUS_OVERHEAD_CHARS = 1500


class ContextManager:
    """Constructs system prompt and manages conversation history window."""

    _CHARS_PER_TOKEN_ESTIMATE: int = 4
    COMPACT_RATIO: float = 0.75
    COMPACT_MIN_SAVINGS_RATIO: float = 0.30
    MIN_RECENT_TURNS: int = 3
    MAX_CONSECUTIVE_FAILURES: int = 3
    MAX_TOOL_RESULTS_FULL: int = 20
    TOOL_RESULT_SUMMARY_CHARS: int = 500
    # v0.9.8: Micro-compaction — lightweight in-process truncation of old
    # tool results to prevent context bloat between full LLM compactions.
    # Inspired by Claude Code's microCompact.ts.
    # v1.0.1 (R4-V15): Lower thresholds so KB retrieve results (often
    # 4KB+ each) get truncated sooner.  Previously 30/15 let 6+ full
    # retrieve chunks accumulate in the conversation history, drowning
    # out the current user question and causing the LLM to produce
    # "summary" responses about prior topics instead of answering the
    # current question.  5/2 keeps the current + previous retrieve
    # intact and reduces older ones to a 200-char extract.
    MICRO_COMPACT_MAX_TOOL_RESULTS: int = 5
    MICRO_COMPACT_KEEP_RECENT: int = 2
    MICRO_COMPACT_TRUNCATE_CHARS: int = 200
    ANCHOR_RULES: list[str] = []
    _skill_dir: Path | None = None
    mcp_status_note: str = ""

    def __init__(self, spec: dict | Protocol):
        """Initialize with a spec dict or any object that satisfies SpecProtocol.

        spec is expected to have the structure:
        - identity: {id, display_name, version}
        - intent: {summary, triggers...}
        - instructions: {workflow, rules, safety, output_template...}
        - interface: {provides, requires, mcp_servers...}
        - resources: {scripts, references...}
        """
        # Store the raw spec for access (supports dict or object with attributes)
        self._raw_spec = spec

        self.history: list[dict[str, Any]] = []
        self.max_history_turns = 40

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
        self._memory: Any = None  # v0.7.6: MemoryBrick reference
        self._kb_system_prompt: str = ""  # v1.0.0: KnowledgeBaseBrick prompt
        self._workflow_note: str = ""  # v0.7.6: current step note from CompiledWorkflow

        self._apply_compact_config()

    @property
    def spec(self) -> Any:
        """Backward-compatible alias for _raw_spec."""
        return self._raw_spec

    def _detect_latest_user_language(self) -> str:
        """Detect the language of the most recent user message in history."""
        for msg in reversed(self.history):
            if msg.get("role") == "user":
                content = msg.get("content", "")
                if isinstance(content, str) and content.strip():
                    cjk_count = sum(1 for c in content if '一' <= c <= '鿿')
                    total = len(content.strip())
                    if total > 0 and cjk_count / total > 0.3:
                        return "Chinese (中文)"
                break
        return "English"

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

    def _get_spec_attr(self, attr_name: str, default: Any) -> Any:
        """Get attribute from raw spec, handling both dict and object."""
        if isinstance(self._raw_spec, dict):
            return self._raw_spec.get(attr_name, default)
        return getattr(self._raw_spec, attr_name, default)

    def build_system_prompt(self, rich: bool = False, _consume_note: bool = True) -> str:
        """Build system prompt from spec with domain persona injection.

        Args:
            rich: Whether to include rich markup (TUI mode).
            _consume_note: Internal flag to control one-shot _workflow_note
                consumption. Set False for token estimation to avoid
                consuming the note before the real prompt is built (C5 fix).
        """
        parts: list[str] = []

        identity = self._get_spec_attr("identity", {})
        intent = self._get_spec_attr("intent", {})
        interface = self._get_spec_attr("interface", {})
        instructions = self._get_spec_attr("instructions", {})
        resources = self._get_spec_attr("resources", {})

        display_name = (
            identity.get("display_name", identity)
            if isinstance(identity, dict)
            else getattr(identity, "display_name", "Agent")
        )

        # v0.7.2: Language directive placed FIRST for highest attention priority.
        # DeepSeek V4 Pro reasoning models deprioritize late-prompt instructions.
        detected_lang = self._detect_latest_user_language()
        if detected_lang and detected_lang != "English":
            parts.append(
                f"CRITICAL INSTRUCTION: You MUST respond in {detected_lang}. "
                f"The user's most recent message was in {detected_lang}. "
                f"Write all your responses in {detected_lang}. "
                f"Do NOT switch to English or any other language."
            )
        else:
            parts.append(
                "CRITICAL INSTRUCTION: Respond in the same language as the user. "
                "Detect the user's language and match it exactly."
            )

        parts.append("")
        parts.append(
            f"You are {display_name}, a specialist agent created by agenthatch."
        )
        if isinstance(identity, dict):
            if identity.get("author"):
                parts.append(f"Author: {identity['author']}")
        elif hasattr(identity, "author") and identity.author:
            parts.append(f"Author: {identity.author}")

        parts.append("")
        parts.append("## Your Core Identity")
        summary = (
            intent.get("summary", "")
            if isinstance(intent, dict)
            else getattr(intent, "summary", "")
        )
        if summary:
            parts.append(f"{summary}")
        parts.append("")
        parts.append(
            "You are a focused, capable specialist. "
            "Always help the user within your expertise. "
            "Use your tools proactively. Give direct, actionable answers. "
            "If a request is truly outside your domain, briefly explain "
            "what you CAN help with instead."
        )
        triggers = (
            intent.get("triggers", [])
            if isinstance(intent, dict)
            else getattr(intent, "triggers", [])
        )
        if triggers:
            parts.append(
                f"You respond to: {', '.join(triggers[:8])}"
                f"{'...' if len(triggers) > 8 else ''}"
            )

        provides = []
        if isinstance(interface, dict):
            provides = [p.get("capability", p) for p in interface.get("provides", [])]
        elif hasattr(interface, "provides"):
            provides = [
                p.capability if hasattr(p, "capability") else str(p)
                for p in interface.provides
            ]
        if provides:
            parts.append(
                f"You have these capabilities: {', '.join(provides)}."
            )
        parts.append(
            "If asked something clearly outside your domain, "
            "politely decline and suggest what you CAN help with."
        )

        # External tool summary (ported from agent/context.py:241-255)
        if self._capbus is not None:
            external_tools = []
            for name, cap in self._capbus.capabilities.items():
                if hasattr(cap, 'type') and cap.type == "external":
                    desc = cap.schema.get("description", name) if hasattr(cap, 'schema') else name
                    external_tools.append(f"- {name}: {desc}")
            if external_tools:
                parts.append("")
                parts.append("## Available External Tools")
                parts.extend(external_tools)

        # v0.7.6: Inject persistent memory into system prompt
        if self._memory is not None:
            memory_section = self._memory.inject_into_context(max_tokens=1000)
            if memory_section:
                parts.append("")
                parts.append("## Agent Memory (from previous sessions)")
                parts.append(memory_section)
                parts.append("")
                parts.append(
                    "Use the `recall` tool to search your memory for "
                    "specific information from past sessions."
                )

        # v1.0.0: Inject KnowledgeBaseBrick section into system prompt
        # v1.0.1 (L5): The injected prompt is now FULL_SYSTEM_PROMPT
        # (B4 + B3 composed).  Strip whitespace and gate on non-empty
        # to avoid emitting a stray "## Knowledge Base" header when
        # the LLM produced an empty SYSTEM_PROMPT_SECTION AND no
        # WHEN_TO_RETRIEVE / QUERY_TEMPLATES were inferred.
        # v1.0.1 (R3-H2): Hard-cap the KB prompt to a fixed char budget
        # so a runaway B4 generation (10KB+ of LLM prose) can't push
        # the system prompt past the model's context window.  8KB ≈
        # 2K tokens, leaving ample room for the rest of the prompt
        # and conversation history.  Truncation is logged so users
        # can spot misbehaving B4 outputs.
        _KB_PROMPT_MAX_CHARS: int = 8192
        kb_prompt = (self._kb_system_prompt or "").strip()
        if kb_prompt:
            if len(kb_prompt) > _KB_PROMPT_MAX_CHARS:
                logger.warning(
                    "ContextManager: KB prompt %d chars exceeds %d cap "
                    "— truncating. Check B4 prompt generation output.",
                    len(kb_prompt), _KB_PROMPT_MAX_CHARS,
                )
                kb_prompt = (
                    kb_prompt[:_KB_PROMPT_MAX_CHARS]
                    + "\n\n...(KB prompt truncated due to size cap)"
                )
            parts.append("")
            parts.append("## Knowledge Base")
            parts.append(kb_prompt)
            parts.append("")
            parts.append(
                "Use the `retrieve` tool to query the knowledge base "
                "before answering questions it may cover."
            )

        # Language directive is now at the TOP of the system prompt (see above)

        workflow = (
            instructions.get("workflow", "")
            if isinstance(instructions, dict)
            else getattr(instructions, "workflow", "")
        )
        if workflow:
            parts.append("\n## Workflow")
            parts.append(
                "Follow these steps in order. Steps marked with "
                "-> Use tool require calling the specified tool."
            )
            # workflow can be a string or list of dicts with step/description
            if isinstance(workflow, list):
                for step in workflow:
                    if isinstance(step, dict):
                        line = f"{step.get('step', '')}. {step.get('description', '')}"
                        if step.get("script"):
                            line += (
                                f"\n   -> Use tool: run_skill_script("
                                f'script_name="{step["script"]}", ...)'
                            )
                        parts.append(line)
                    elif hasattr(step, "step") and hasattr(step, "description"):
                        line = f"{step.step}. {step.description}"
                        if hasattr(step, "script") and step.script:
                            line += (
                                f"\n   -> Use tool: run_skill_script("
                                f'script_name="{step.script}", ...)'
                            )
                        parts.append(line)
            else:
                parts.append(workflow)

        # v0.7.6: Runtime workflow step note from CompiledWorkflow._pre_turn_workflow()
        # This overrides the static workflow text with the CURRENT step
        if self._workflow_note:
            parts.append(f"\n## Current Step\n{self._workflow_note}")
            if _consume_note:
                self._workflow_note = ""  # one-shot: consumed, then cleared

        rules = (
            instructions.get("rules", [])
            if isinstance(instructions, dict)
            else getattr(instructions, "rules", [])
        )
        if rules:
            parts.append("\n## Rules - NEVER violate these")
            for rule in rules:
                parts.append(f"- {rule}")

        safety = (
            instructions.get("safety", {})
            if isinstance(instructions, dict)
            else getattr(instructions, "safety", {})
        )
        if (
            (isinstance(safety, dict) and safety.get("plan_required"))
            or (hasattr(safety, "plan_required") and safety.plan_required)
        ):
            parts.append("\nAlways create a plan before executing.")

        raw_body = (
            instructions.get("raw_body", "")
            if isinstance(instructions, dict)
            else getattr(instructions, "raw_body", "")
        )
        if raw_body:
            if len(raw_body) > 3000:
                raw_body = raw_body[:3000] + "\n\n... (truncated for context window)"
            parts.append("\n## Operational Guidance")
            parts.append(raw_body)

        references = (
            resources.get("references", [])
            if isinstance(resources, dict)
            else getattr(resources, "references", [])
        )
        if references:
            parts.append("\n## Reference Documents")
            parts.append(
                "These reference documents are available for domain knowledge:"
            )
            for ref in references:
                ref_name = ref.get("name", ref.get("path", "unknown"))
                parts.append(f"- {ref_name}")

        scripts = (
            resources.get("scripts", [])
            if isinstance(resources, dict)
            else getattr(resources, "scripts", [])
        )
        if scripts:
            parts.append("\n## Available Scripts")
            parts.append(
                "These scripts are available via run_skill_script(). "
                "Call them by script_name."
            )
            for script in scripts:
                script_name = script.get("name", "")
                parts.append(f"- run_skill_script(script_name=\"{script_name}\")")

        output_template = (
            instructions.get("output_template", "")
            if isinstance(instructions, dict)
            else getattr(instructions, "output_template", "")
        )
        if output_template:
            parts.append("")
            parts.append(
                "## Output Format (MANDATORY — enforced every turn)"
            )
            if self.summary is not None:
                parts.append(
                    "This is a long-running session. "
                    "You MUST still follow the output template below. "
                    "Do not drift into free-form text."
                )
            parts.append(f"\n{output_template}")

        if self.ANCHOR_RULES and (
            self.summary is not None or (self._turn_count > 0 and self._turn_count % 20 == 0)
        ):
            parts.append("\n## Core Rules (NEVER forget)")
            for rule in self.ANCHOR_RULES:
                parts.append(f"- {rule}")

        if self._batch_total > 0:
            parts.append("")
            parts.append("## Progress Tracking")
            parts.append(
                f"You are processing a batch of {self._batch_total} items. "
                f"Current progress: {self._batch_current}/{self._batch_total}. "
                "After completing each item, clearly state your progress "
                "and continue to the next item. Stop when all items are done."
            )

        if self.mcp_status_note:
            parts.append("")
            parts.append(self.mcp_status_note)

        return "\n".join(parts)

    def build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """Build complete message list (system + history + user)."""
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
        # v0.9.8: Micro-compact after each tool result to prevent
        # unbounded context growth between full LLM compactions.
        if role == "tool":
            self.micro_compact()

    def micro_compact(self) -> int:
        """Lightweight in-process truncation of old tool results.

        Unlike full compact() which calls an LLM, this is a deterministic
        truncation that keeps recent tool results intact and summarizes
        older ones to a brief extract.  Preserves tool_call_id → tool_result
        pairing integrity.

        Returns the number of tool results truncated.
        """
        # Count tool results and find their positions
        tool_indices: list[int] = []
        for i, msg in enumerate(self.history):
            if msg.get("role") == "tool":
                tool_indices.append(i)

        total_tools = len(tool_indices)
        if total_tools <= self.MICRO_COMPACT_MAX_TOOL_RESULTS:
            return 0

        # Keep the most recent MICRO_COMPACT_KEEP_RECENT tool results intact
        keep_from = tool_indices[-self.MICRO_COMPACT_KEEP_RECENT] if len(tool_indices) >= self.MICRO_COMPACT_KEEP_RECENT else 0

        truncated = 0
        for idx in tool_indices:
            if idx >= keep_from:
                break  # reached the "keep recent" zone
            content = self.history[idx].get("content", "")
            if isinstance(content, str) and len(content) > self.MICRO_COMPACT_TRUNCATE_CHARS:
                self.history[idx]["content"] = (
                    content[:self.MICRO_COMPACT_TRUNCATE_CHARS]
                    + "... [micro-compacted]"
                )
                truncated += 1

        if truncated > 0:
            logger.debug(
                "Micro-compact: truncated %d old tool results "
                "(total: %d, kept recent: %d)",
                truncated, total_tools,
                min(self.MICRO_COMPACT_KEEP_RECENT, total_tools),
            )
        return truncated

    def estimate_input_tokens(self) -> int:
        """Estimate total input tokens for current context."""
        # Pass _consume_note=False to avoid consuming the one-shot
        # _workflow_note during token estimation (C5 fix).
        system_len = len(self.build_system_prompt(_consume_note=False))
        history_len = sum(
            len(str(msg.get("content", "") or ""))
            + len(str(msg.get("tool_calls", "")))
            for msg in self.history
        )
        return (system_len + history_len + CAPBUS_OVERHEAD_CHARS) // self._CHARS_PER_TOKEN_ESTIMATE

    def auto_compact_check(self, max_tokens: int) -> bool:
        """Check if compaction should be triggered and execute it."""
        if self._consecutive_compact_failures >= self.MAX_CONSECUTIVE_FAILURES:
            logger.warning(
                "Circuit breaker open: %d consecutive compact failures. Skipping.",
                self._consecutive_compact_failures,
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

        if self._hooks:
            self._hooks.execute(HookPoint.POST_COMPACT, hook_ctx)

        return result

    def _should_compact(self, max_tokens: int, current_tokens: int) -> bool:
        """Check if compaction is worth the token cost."""
        threshold = int(max_tokens * self.COMPACT_RATIO)
        if current_tokens < threshold:
            return False

        recent_tokens = sum(
            len(str(m.get("content", ""))) // self._CHARS_PER_TOKEN_ESTIMATE
            for m in self.history[-(self.MIN_RECENT_TURNS * 2):]
        )
        estimated_after = 500 + recent_tokens
        savings_ratio = 1.0 - (estimated_after / current_tokens)

        if savings_ratio < self.COMPACT_MIN_SAVINGS_RATIO:
            logger.info(
                "Skipping compaction: estimated savings %.1f%% < %.1f%% minimum",
                savings_ratio * 100, self.COMPACT_MIN_SAVINGS_RATIO * 100,
            )
            return False

        return True

    def compact(self, max_tokens: int) -> bool:
        """Execute compaction. Returns True on success."""
        try:
            summary = self._generate_summary()
        except Exception as e:
            # C5/H8 fix: catch all exceptions from LLM calls
            # (network errors, timeouts, etc.), not just JSON/Key/Value errors
            logger.warning("Compact failed (%s: %s), falling back to truncation",
                           type(e).__name__, e)
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
        """Truncate history to recent turns."""
        keep_count = self.MIN_RECENT_TURNS * 2

        if len(self.history) <= keep_count:
            return False

        safe_boundary = len(self.history) - keep_count
        for i in range(safe_boundary - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                safe_boundary = i
                break
            elif msg.get("role") == "tool":
                continue
            else:
                break

        tool_call_ids: set[str] = set()
        for i in range(safe_boundary, len(self.history)):
            msg = self.history[i]
            if msg.get("role") == "assistant" and msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    if tc.get("id"):
                        tool_call_ids.add(tc["id"])

        for i in range(safe_boundary - 1, -1, -1):
            msg = self.history[i]
            if msg.get("role") == "tool":
                tid = msg.get("tool_call_id", "")
                if tid in tool_call_ids:
                    # Move boundary to include the parent assistant+tool_calls
                    for j in range(i - 1, -1, -1):
                        parent = self.history[j]
                        if parent.get("role") == "assistant" and parent.get("tool_calls"):
                            if any(tc.get("id") == tid for tc in parent["tool_calls"]):
                                safe_boundary = j
                                break
                    break

        self.history = self.history[safe_boundary:]
        logger.debug(
            "Truncated history to %d messages (preserved tool call chains)",
            len(self.history),
        )
        return False

    def _offload_full_history(self) -> Path | None:
        """Offload full conversation history before truncation."""
        if self._state_manager is None:
            return None
        return self._state_manager.save_history(list(self.history))

    def _build_compact_messages(self) -> list[dict[str, Any]]:
        """Build messages for the compaction LLM call.

        v0.9.8: Include tool call results (truncated) so the LLM knows
        what tools were called and what happened.  Previously only user
        and assistant (non-tool-call) messages were included, making the
        tool_calls_summary field in CompactSummary always empty.
        """
        prompt = COMPACT_SYSTEM_PROMPT

        if self.summary:
            prompt += "\n\n## Prior Compaction Summary\n"
            prompt += self.summary.to_text()
            prompt += "\n\nUpdate this summary to include the recent conversation below."

        if self.ANCHOR_RULES:
            prompt += (
                "\n\nIMPORTANT: Preserve these rules in your summary: "
                + "; ".join(self.ANCHOR_RULES)
            )

        prompt += "\n\n## Conversation to Compact\n"

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt},
        ]

        # Max chars for a single tool result in the compaction prompt.
        # Tool results are truncated to this length to avoid token bloat
        # while still giving the LLM enough context to summarize.
        TOOL_COMPACT_CHARS = 800

        for msg in self.history:
            role = msg["role"]
            if role in ("user", "assistant"):
                if role == "assistant" and msg.get("tool_calls"):
                    # Include tool call requests (function names + arguments)
                    # but strip the full content to save tokens
                    tc_msg = dict(msg)
                    tc_msg.pop("content", None)
                    messages.append(tc_msg)
                elif role == "assistant":
                    messages.append(msg)
                elif role == "user":
                    messages.append(msg)
            elif role == "tool":
                # Include tool results, truncated to TOOL_COMPACT_CHARS
                content = msg.get("content", "")
                if isinstance(content, str) and len(content) > TOOL_COMPACT_CHARS:
                    tc_msg = dict(msg)
                    tc_msg["content"] = (
                        content[:TOOL_COMPACT_CHARS]
                        + "... [truncated for compaction]"
                    )
                    messages.append(tc_msg)
                else:
                    messages.append(msg)

        return messages

    def _generate_summary(self) -> CompactSummary:
        """Call LLM to generate a CompactSummary from conversation history.

        Tries structured output first (Instructor/JSON mode).  Falls back to
        raw chat with balanced-JSON extraction when structured mode fails
        (common with reasoning models like deepseek-v4-pro).
        """
        messages = self._build_compact_messages()

        if self._llm is None:
            raise RuntimeError("No LLM client available for compaction")

        try:
            result = self._llm.chat_structured(
                messages=messages,
                response_model=CompactSummary,
                max_retries=2,
            )
            return result  # type: ignore[no-any-return]
        except Exception as e:
            logger.warning(
                "Structured compaction failed (%s), falling back to raw chat", e
            )

        # Fallback: raw chat with JSON extraction
        try:
            raw = self._llm.chat(messages=messages)
            json_candidates = _extract_balanced_json(raw)
            if json_candidates:
                for candidate in json_candidates:
                    try:
                        data = json.loads(candidate)
                        return CompactSummary(**data)
                    except Exception:
                        continue
            if json_candidates:
                logger.warning(
                    "_generate_summary(): all %d JSON candidates failed, "
                    "using raw text fallback (quality degraded)",
                    len(json_candidates),
                )
            # Minimal fallback: use the raw text as session_intent
            return CompactSummary(
                session_intent=raw[:500],
                current_state="compact_fallback",
            )
        except Exception as e2:
            logger.error("Raw chat compaction also failed: %s", e2)
            return CompactSummary(
                session_intent="compaction failed",
                current_state="error",
            )

    def compress(self, rich: bool = False) -> str:
        """DEPRECATED: Stub for backward compatibility.

        Context compression is now handled by agent_loop.py.
        Returns a notice that compression is managed by the conversation loop.
        """
        return "Context compression is not yet available; managed by ConversationLoop."

    def set_batch_scope(self, total: int) -> None:
        """Set batch scope for progress tracking."""
        self._batch_total = total
        self._batch_current = 0

    def advance_batch(self) -> None:
        """Advance batch counter."""
        self._batch_current += 1
