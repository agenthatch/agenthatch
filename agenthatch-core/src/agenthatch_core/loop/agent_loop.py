"""ConversationLoop — LLM <-> Tool calling cycle (agenthatch-core).

Ports features from agenthatch/agent/loop.py: hooks, token_counter, parallel
execution, context_window awareness, reasoning content handling.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import re
import time
from collections.abc import Callable, Generator
from typing import Any, cast

from agenthatch_core.context.manager import ContextManager
from agenthatch_core.exceptions import CapabilityNotFoundError
from agenthatch_core.hooks import HookPoint, HooksManager
from agenthatch_core.llm.client import LLMClient, ToolCallResponse
from agenthatch_core.loop.token_counter import TokenCounter, ThinkingDelta
from agenthatch_core.sandbox.executor import Sandbox
from agenthatch_core.tools.bus import CapBus

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 10000

# v0.9: Interrupt message injected when user interrupts agent mid-execution.
_INTERRUPT_MESSAGE = (
    "User interrupted. Stop what you are doing, ask the user what they "
    "want next, and wait for their response. Do not continue executing "
    "tools or auto-continuing."
)

# ── v0.6 Autonomous task completion ──────────────────────────────────
_TASK_COMPLETE_TOOL = "task_complete"
_CONTINUE_NUDGE = (
    "Continue working on the user's request if there are remaining steps. "
    "Call task_complete ONLY when you have completed ALL of the user's "
    "requests, not after each individual sub-task."
)
# Grace period: allow N text-only responses before nudging,
# so the agent can report progress naturally without being
# pushed to task_complete prematurely.
_NUDGE_GRACE = 2


def _route_with_timeout(
    capbus: CapBus, tool_name: str, arguments: dict[str, Any], timeout: int = 120
) -> str:
    """Execute tool with timeout to prevent infinite hangs.

    Uses explicit executor management with shutdown(wait=False) because
    Python threads cannot be killed — if the tool thread is stuck in I/O,
    shutdown(wait=True) would deadlock the entire agent conversation.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(capbus.route, tool_name, arguments)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return f"Error: tool '{tool_name}' timed out after {timeout}s"
    finally:
        executor.shutdown(wait=False)


class RichToolCallEvent:
    """Rich-renderable tool call event for TUI consumption."""

    def __init__(
        self,
        phase: str,
        tool_name: str,
        tool_args: dict[str, Any] | None = None,
        elapsed: float | None = None,
        result_preview: str | None = None,
    ):
        self.phase = phase
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.elapsed = elapsed
        self.result_preview = result_preview


# v0.8.15: Minimal safety net — consecutive empty-text responses before breaking.
# Prevents tight infinite loops where the LLM returns text without tool calls
# repeatedly.  No token budget, no round limit.  The agent runs freely.
_MAX_CONSECUTIVE_TEXT_ONLY = 13

# v1.0.1 (R4-V21): Trailing meta-narration patterns for KB agents.
# Despite B4 (e) explicitly forbidding meta-commentary, the LLM often
# appends sentences like "已完整解答…无剩余步骤" or "用户的问题…已在
# 上一轮完整解答" before calling task_complete.  These patterns are
# deterministic enough to strip safely at the code layer.
#
# Each pattern matches a phrase that typically begins a meta-narration
# sentence.  We find the earliest such pattern in the last 600 chars
# and strip everything from that point to the end of the text.
#
# Safety: stripping is aborted if it would remove more than 40% of
# the response (guard against false positives on short answers).
_TRAILING_META_NARRATION_PATTERNS: tuple[str, ...] = (
    # "用户的问题/提问/请求...已..."
    r"用户.{0,12}(?:问题|提问|请求).{0,40}已",
    # "已完整/逐一...解答/说明/呈现/呈上/作答"
    r"已(?:完整|逐一|详细).{0,4}(?:解答|说明|呈现|呈上|作答)",
    # "前已..." (前已完整作答 / 前已答 / 前已说明)
    r"前已(?:完整|逐一|详细?).{0,4}(?:作答|解答|说明|呈现|呈上|答)",
    # v1.0.1 (R4-V22): "前问已答毕" / "前问已答" — LLM 变体
    r"前问已答",
    # v1.0.1 (R4-V22): "已答毕" — 通用 meta-narration 闭合
    r"已答毕",
    # "无/没有剩余步骤/任务"
    r"(?:无|没有)剩余.{0,4}(?:步骤|任务)",
    # "任务已完成"
    r"任务已完成",
    # "当前无剩余"
    r"当前无剩余",
    # "无需继续处理"
    r"无需继续处理",
    # "本次回答完毕" / "回答完毕"
    r"(?:本次)?回答完毕",
    # "所有问题均已..."
    r"所有问题均已",
    # "已为您解答"
    r"已为您解答",
    # "已在上轮" / "已在前一轮"
    r"已在前?一轮",
    # "没有未完成的步骤"
    r"没有未完成的步骤",
    # "已逐一解答"
    r"已逐一",
    # v1.0.1 (R4-V22): "前文已..." / "上文已..." — 多轮历史污染
    r"前文已(?:完整|逐一|详细)?.{0,4}(?:作答|解答|说明|呈现|呈上|答)",
    r"上文已(?:完整|逐一|详细)?.{0,4}(?:作答|解答|说明|呈现|呈上|答)",
    # v1.0.1 (R4-V22): "已详答" / "已作答" — 简短变体
    r"已详答",
    r"已作答",
    # English: "The request has been fully addressed"
    r"[Tt]he (?:request|question|user).{0,40}(?:addressed|delivered|answered|resolved)",
    # English: "has been fully delivered/addressed"
    r"has been (?:fully )?(?:delivered|addressed|resolved)",
    # English: "No remaining steps"
    r"[Nn]o remaining steps",
    # English: "Task complete" / "Task completed"
    r"[Tt]ask (?:is )?(?:complete|completed|done)",
)


def _strip_trailing_meta_narration(text: str) -> str:
    """Strip trailing meta-narration from a KB agent response.

    v1.0.1 (R4-V21): The LLM often appends meta-commentary before
    calling task_complete, despite B4 (e) explicitly forbidding it.
    This function finds meta-narration patterns in the last 600 chars
    and removes the *entire sentence* containing each match — so
    meta-narration embedded mid-response (with real content after it)
    is removed while the surrounding content is preserved.

    v1.0.1 (R4-V22): Rewrote from "truncate at earliest match" to
    "delete whole sentence" — the truncate approach removed real
    content that followed the meta-narration (e.g. the closing
    "阁下若欲探询…" sentence), and the 40% safety guard would then
    abort stripping entirely, leaking the meta-narration through.

    Safety: stripping is aborted if it would remove more than 50% of
    the response (guard against false positives on short answers).
    """
    if not text or len(text) < 20:
        return text

    # Search for meta-narration patterns in the last 600 chars
    tail_start = max(0, len(text) - 600)
    tail = text[tail_start:]

    # Find all matches with their absolute positions
    matches: list[tuple[int, int]] = []
    for pat in _TRAILING_META_NARRATION_PATTERNS:
        for m in re.finditer(pat, tail):
            matches.append((tail_start + m.start(), tail_start + m.end()))

    if not matches:
        return text

    # Expand each match to the full sentence boundary
    # (sentence terminators: 。！？\n)
    _TERMINATORS = set("。！？\n")
    ranges_to_delete: list[tuple[int, int]] = []
    for abs_start, abs_end in matches:
        # Walk backward to find sentence start
        sent_start = 0
        for i in range(abs_start - 1, -1, -1):
            if text[i] in _TERMINATORS:
                sent_start = i + 1
                break
        # Walk forward to find sentence end (inclusive of terminator)
        sent_end = len(text)
        for i in range(abs_end, len(text)):
            if text[i] in _TERMINATORS:
                sent_end = i + 1
                break
        ranges_to_delete.append((sent_start, sent_end))

    # Merge overlapping ranges
    ranges_to_delete.sort()
    merged: list[tuple[int, int]] = []
    for s, e in ranges_to_delete:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # Build result by skipping deleted ranges
    parts: list[str] = []
    last = 0
    for s, e in merged:
        parts.append(text[last:s])
        last = e
    parts.append(text[last:])
    stripped = "".join(parts).rstrip()

    # Safety: don't strip if it removes more than 50% of the text
    if len(stripped) >= max(1, int(len(text) * 0.5)):
        return stripped
    return text


class ConversationLoop:
    """Drives the User -> LLM -> Tool -> LLM -> Response cycle.

    v0.8.15: No artificial limits.  The agent runs until the task is
    naturally complete.  The only guard is _MAX_CONSECUTIVE_TEXT_ONLY
    to prevent tight infinite auto-continuation loops.
    """

    def __init__(
        self,
        llm: LLMClient,
        capbus: CapBus,
        sandbox: Sandbox,
        ctx: ContextManager,
        hooks: HooksManager | None = None,
        token_counter: TokenCounter | None = None,
        memory_brick: Any = None,  # v0.7.11: MemoryBrick for persistent memory
        checkpoint_mgr: Any = None,  # v0.7.12: CheckpointManager for conversation persistence
        plan_layer: Any = None,  # v0.9.8: PlanLayer for state-machine planning
        max_consecutive_text_only: int | None = None,  # v1.0.1 (R4-V17)
        nudge_grace: int | None = None,  # v1.0.1 (R4-V17)
    ):
        self.llm = llm
        self.capbus = capbus
        self.sandbox = sandbox
        self.ctx = ctx
        self._hooks = hooks
        self._token_counter = token_counter
        self._memory_brick = memory_brick  # v0.7.11
        self._checkpoint_mgr = checkpoint_mgr  # v0.7.12: FROM PARAMETER
        self._plan_layer = plan_layer  # v0.9.8: PlanLayer for plan-guided agents

        # v1.0.1 (R4-V17): Per-agent auto-continuation tuning.
        # KB agents retrieve once and produce a single answer; the default
        # _MAX_CONSECUTIVE_TEXT_ONLY=13 + _NUDGE_GRACE=2 lets the loop
        # auto-continue after the first text response, producing duplicate
        # answers and meta-summaries ("已回答…").  When max_consecutive_text_only
        # is passed (e.g. 1 for KB agents), the loop breaks after the first
        # text-only response following tool execution, returning the answer
        # immediately.
        self._max_consecutive_text_only: int = (
            max_consecutive_text_only
            if max_consecutive_text_only is not None
            else _MAX_CONSECUTIVE_TEXT_ONLY
        )
        self._nudge_grace: int = (
            nudge_grace if nudge_grace is not None else _NUDGE_GRACE
        )

        # v0.9: Interruptable execution — set by EarlyInputReader on Ctrl+C
        self._interrupted = False

        self._max_retries: int = 3
        self._retry_base_delay: float = 1.0
        self._retry_max_delay: float = 30.0
        self._retryable_statuses: set[int] = {429, 500, 502, 503, 504}

        self._cb_threshold: int = 5
        self._cb_timeout: float = 60.0
        self._cb_failures: int = 0
        self._cb_state: str = "closed"
        self._cb_opened_at: float = 0.0

    # ── v0.9: Interrupt check ─────────────────────────────────────────

    def _check_interrupted(self) -> bool:
        """Check if user interrupted execution (Ctrl+C during streaming).

        Also checks ctx._interrupted which may be set via signal handler.
        Returns True if execution should stop immediately.
        """
        if self._interrupted:
            return True
        if getattr(self.ctx, "_interrupted", False):
            self._interrupted = True
            return True
        return False

    def _record_usage(self, response: Any) -> None:
        """Record token usage from LLM response into token counter."""
        if self._token_counter is None:
            return
        usage = getattr(response, "usage", None)
        # Fall back to LLM client's last_usage if response lacks usage
        if usage is None:
            usage = getattr(self.llm, "last_usage", None)
        if usage is None:
            # v0.7.12: CJK-aware token estimation fallback
            # DeepSeek streaming doesn't return usage in chunk events.
            # Estimate tokens from response text content.
            text = getattr(response, "text", "") or ""
            if text:
                cjk_count = sum(
                    1 for c in text
                    if '\u4e00' <= c <= '\u9fff'
                    or '\u3000' <= c <= '\u303f'
                )
                other_count = len(text) - cjk_count
                estimated = max(1, cjk_count + other_count // 4)
                self._token_counter.add_usage({
                    "prompt_tokens": 0,
                    "completion_tokens": estimated,
                    "total_tokens": estimated,
                    "reasoning_tokens": 0,
                })
            return
        self._token_counter.add_usage({
            "prompt_tokens": getattr(usage, "prompt_tokens", 0),
            "completion_tokens": getattr(usage, "completion_tokens", 0),
            "total_tokens": getattr(usage, "total_tokens", 0),
            "reasoning_tokens": getattr(usage, "reasoning_tokens", 0),
            "completion_tokens_details": getattr(usage, "completion_tokens_details", None) or {},
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", 0),
            "cached_tokens": (
                getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0)
                or 0
            ),
        })

    def run(self, user_input: str) -> str:
        """Execute one conversation turn synchronously."""
        self.ctx._turn_count += 1

        # PRE_TURN hook
        if self._hooks:
            turn_ctx: dict[str, Any] = {
                "user_input": user_input,
                "turn_count": self.ctx._turn_count,
            }
            turn_ctx = self._hooks.execute(HookPoint.PRE_TURN, turn_ctx)

        self.ctx.auto_compact_check(self.llm.context_window)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        # v0.9.8: PlanLayer — inject plan context into system prompt
        if self._plan_layer is not None:
            self._plan_layer.handle_turn_start()
            plan_ctx = self._plan_layer.plan_context
            if plan_ctx and messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + plan_ctx
            suggestion = self._plan_layer.next_suggestion
            if suggestion and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + suggestion

        tools_for_api = [
            {"type": t.type, "function": t.function}
            if hasattr(t, "type") else
            {"type": t["type"], "function": t["function"]}
            for t in tools
        ]

        if not self._cb_allow():
            return "Service temporarily unavailable. Please wait and try again."

        try:
            response = self._call_with_retry(
                self.llm.chat_with_tools, messages=messages, tools=tools_for_api,
            )
            self._cb_record(True)
            self._record_usage(response)
        except Exception as e:
            self._cb_record(False)
            logger.error("LLM API call failed: %s", e)
            return f"I encountered an error communicating with the model provider: {e}"

        has_executed_tools = False
        tool_stats: dict[str, int] = {}
        _consecutive_text_only = 0  # v0.8.15: prevent infinite auto-continuation
        while True:
            # v0.6: detect task_complete signal, return summary
            if response.tool_calls:
                tc_names = [tc.name for tc in response.tool_calls]
                if _TASK_COMPLETE_TOOL in tc_names:
                    work_tools = [tc for tc in response.tool_calls
                                  if tc.name != _TASK_COMPLETE_TOOL]
                    if not work_tools:
                        summary = response.tool_calls[
                            tc_names.index(_TASK_COMPLETE_TOOL)
                        ].arguments.get("summary", "Done.")
                        # v1.0.1 (R4-V20): For KB agents, the real answer
                        # is in response.text (the LLM emits it alongside
                        # the task_complete call).  Use response.text when
                        # available; fall back to the task_complete summary.
                        # Previously the summary was always returned, which
                        # for KB agents was a meta-summary like "已回答…"
                        # rather than the substantive answer.
                        final_text = response.text or summary
                        # v1.0.1 (R4-V21): Strip trailing meta-narration
                        # for KB agents.  The LLM often appends sentences
                        # like "已完整解答…无剩余步骤" before calling
                        # task_complete, despite B4 (e) forbidding it.
                        if self._max_consecutive_text_only == 0:
                            final_text = _strip_trailing_meta_narration(final_text)
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", final_text)
                        # v0.7.11: Record turn to memory
                        if self._memory_brick:
                            self._memory_brick.record_turn("user", user_input)
                            self._memory_brick.record_turn("assistant", final_text)
                        if self._hooks:
                            _ = self._hooks.execute(HookPoint.POST_TURN, {
                                "turn_count": self.ctx._turn_count,
                                "final_text": final_text,
                            })
                        self._checkpoint()
                        return cast("str", final_text)
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.tool_calls:
                # v0.6: Auto-continuation only after tools executed
                # (needsFollowUp pattern)
                if not has_executed_tools:
                    break
                # v0.8.14: Limit consecutive text-only to prevent
                # infinite auto-continuation loops.
                _consecutive_text_only += 1
                if _consecutive_text_only > self._max_consecutive_text_only:
                    break
                # v0.9.6: Grace period — allow the agent to report
                # progress naturally (e.g. "Page opened, search box visible")
                # before nudging it to continue or complete.
                if _consecutive_text_only < self._nudge_grace:
                    messages.append({
                        "role": "assistant",
                        "content": response.text,
                    })
                    self.ctx.add_to_history("assistant", str(response.text))
                    try:
                        response = self._call_with_retry(
                            self.llm.chat_with_tools, messages, tools_for_api,
                        )
                        self._cb_record(True)
                        self._record_usage(response)
                    except Exception as e:
                        self._cb_record(False)
                        logger.error("LLM API call failed in tool loop: %s", e)
                        response = ToolCallResponse(
                            text=f"Error communicating with model provider: {e}",
                            tool_calls=[],
                        )
                    continue
                messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                })
                self.ctx.add_to_history("assistant", response.text)
                messages.append({"role": "user", "content": _CONTINUE_NUDGE})

                # v0.9: Check interrupt before auto-continuation
                if self._check_interrupted():
                    self.ctx.add_to_history("user", user_input)
                    return _INTERRUPT_MESSAGE

                if not self._cb_allow():
                    break
                try:
                    response = self._call_with_retry(
                        self.llm.chat_with_tools, messages, tools_for_api,
                    )
                    self._cb_record(True)
                    self._record_usage(response)
                except Exception as e:
                    self._cb_record(False)
                    logger.error("LLM API call failed in tool loop: %s", e)
                    response = ToolCallResponse(
                        text=f"Error communicating with model provider: {e}",
                        tool_calls=[],
                    )
                continue

            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ],
            }
            # v0.9.8: Preserve reasoning_content for DeepSeek thinking mode
            if response.reasoning_content:
                assistant_msg["reasoning_content"] = response.reasoning_content
            messages.append(assistant_msg)

            self.ctx.add_to_history(
                "assistant",
                None,
                tool_calls=assistant_msg.get("tool_calls"),
            )

            # PRE_TOOL_CALL hook
            if self._hooks and response.tool_calls:
                _ = self._hooks.execute(HookPoint.PRE_TOOL_CALL, {
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                })

            # v0.9: Check interrupt before executing tools
            if self._check_interrupted():
                self.ctx.add_to_history("user", user_input)
                return _INTERRUPT_MESSAGE

            parallel = (
                self.llm.features.supports_parallel_tool_calls
                and len(response.tool_calls) > 1
            )
            # v0.9.8: Collect tool results for PlanLayer
            plan_tool_results: list[dict[str, Any]] = []
            for entry in self._execute_tool_calls(
                response.tool_calls, parallel=parallel
            ):
                tc = entry["tc"]
                elapsed = entry["elapsed"]
                tool_stats[tc.name] = tool_stats.get(tc.name, 0) + 1
                result_str = entry["result"]
                if len(result_str) > MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                self.ctx.add_to_history(
                    "tool",
                    f"[{tc.name}]: {str(result_str)[:500]}",
                    tool_call_id=tc.id,
                )

                # POST_TOOL_CALL hook
                if self._hooks:
                    _ = self._hooks.execute(HookPoint.POST_TOOL_CALL, {
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_str,
                        "elapsed": elapsed,
                    })
                has_executed_tools = True
                _consecutive_text_only = 0  # v0.8.14: reset on tool execution

                # v0.9.8: Collect for PlanLayer
                plan_tool_results.append({
                    "name": tc.name,
                    "content": result_str,
                })

            # v0.9.8: PlanLayer — update state after tool execution
            if self._plan_layer is not None and plan_tool_results:
                self._plan_layer.handle_turn_end(
                    response_text=None,
                    tool_results=plan_tool_results,
                )
                # Re-inject plan context after state change
                if self._plan_layer.state.value != "done":
                    plan_ctx = self._plan_layer.plan_context
                    if plan_ctx and messages and messages[0]["role"] == "system":
                        base = self.ctx.build_system_prompt()
                        messages[0]["content"] = base + "\n" + plan_ctx
                    suggestion = self._plan_layer.next_suggestion
                    if suggestion:
                        messages.append({"role": "user", "content": suggestion})

            if not self._cb_allow():
                break

            try:
                response = self._call_with_retry(
                    self.llm.chat_with_tools, messages, tools_for_api,
                )
                self._cb_record(True)
                self._record_usage(response)
            except Exception as e:
                self._cb_record(False)
                logger.error("LLM API call failed in tool loop: %s", e)
                response = ToolCallResponse(
                    text=f"Error communicating with model provider: {e}",
                    tool_calls=[],
                )

        self.ctx.add_to_history("user", user_input)
        final_text = response.text if response and response.text else ""
        # v1.0.1 (R4-V21): Strip trailing meta-narration for KB agents
        # on the non-task_complete exit path too.
        if self._max_consecutive_text_only == 0:
            final_text = _strip_trailing_meta_narration(final_text)
        if final_text:
            self.ctx.add_to_history("assistant", final_text)
        # v0.7.11: Record turn to memory
        if self._memory_brick:
            self._memory_brick.record_turn("user", user_input)
            if final_text:
                self._memory_brick.record_turn("assistant", final_text)

        # POST_TURN hook
        if self._hooks:
            _ = self._hooks.execute(HookPoint.POST_TURN, {
                "turn_count": self.ctx._turn_count,
                "final_text": final_text,
            })

        self._checkpoint()
        return final_text or "(no response)"

    def _call_with_retry(
        self, fn: Callable[..., Any], *args: Any, **kwargs: Any
    ) -> Any:
        """Call fn with exponential backoff on transient errors."""
        for attempt in range(self._max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                status = (
                    getattr(e, "status_code", None)
                    or getattr(getattr(e, "response", None), "status_code", None)
                    or getattr(e, "code", None)
                )
                if status not in self._retryable_statuses:
                    raise
                if attempt == self._max_retries:
                    raise
                delay = min(
                    self._retry_base_delay * (2 ** attempt),
                    self._retry_max_delay,
                )
                delay *= random.uniform(0.75, 1.25)
                logger.warning(
                    "Retry %d/%d after %.1fs: %s",
                    attempt + 1, self._max_retries, delay, e,
                )
                time.sleep(delay)

    def _cb_allow(self) -> bool:
        """Check if circuit breaker allows the request."""
        if self._cb_state == "closed":
            return True
        if self._cb_state == "open":
            if time.time() - self._cb_opened_at > self._cb_timeout:
                self._cb_state = "half_open"
                logger.info("Circuit breaker: OPEN -> HALF_OPEN")
                return True
            return False
        return True

    def _cb_record(self, success: bool) -> None:
        """Record a request result."""
        if success:
            if self._cb_state == "half_open":
                self._cb_state = "closed"
                self._cb_failures = 0
                logger.info("Circuit breaker: HALF_OPEN -> CLOSED")
            elif self._cb_state == "closed":
                self._cb_failures = 0
        else:
            self._cb_failures += 1
            if (
                self._cb_state == "closed"
                and self._cb_failures >= self._cb_threshold
            ):
                self._cb_state = "open"
                self._cb_opened_at = time.time()
                logger.warning(
                    "Circuit breaker: CLOSED -> OPEN (%d failures)",
                    self._cb_failures,
                )
            elif self._cb_state == "half_open":
                self._cb_state = "open"
                self._cb_opened_at = time.time()
                logger.warning(
                    "Circuit breaker: HALF_OPEN -> OPEN (probe failed)"
                )

    def _execute_tool_calls(
        self,
        tool_calls: list[Any],
        *,
        parallel: bool = False,
    ) -> list[dict[str, Any]]:
        """Execute tool calls, optionally in parallel via ThreadPoolExecutor.

        Returns a list of result dicts in the same order as input tool_calls,
        each with keys: tc, result (str), elapsed (float).
        Single-call or parallel=False uses a fast sequential path.
        """
        if not parallel or len(tool_calls) <= 1:
            results: list[dict[str, Any]] = []
            for tc in tool_calls:
                t0 = time.time()
                logger.info(
                    "  Executing: %s(%s)...", tc.name,
                    ", ".join(
                        f"{k}={v}" for k, v in tc.arguments.items()
                        if k != "url"
                    ),
                )
                try:
                    result = _route_with_timeout(self.capbus, tc.name, tc.arguments)
                    elapsed = time.time() - t0
                    logger.info(
                        "  %s -> %d chars (%.1fs)", tc.name,
                        len(str(result)), elapsed,
                    )
                except CapabilityNotFoundError as e:
                    elapsed = time.time() - t0
                    logger.warning("Tool call failed: %s (%.1fs)", e, elapsed)
                    result = f"Error: {e}"
                except Exception as e:
                    elapsed = time.time() - t0
                    logger.warning("Tool execution failed: %s (%.1fs)", e, elapsed)
                    result = f"Error: {e}"
                results.append({"tc": tc, "result": str(result), "elapsed": elapsed})
            return results

        # v0.7.12: Parallel path with per-call timeout safety.
        # Uses _route_with_timeout() which creates its own executor per call,
        # matching the sequential path's safety guarantees.
        results_by_index: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(tool_calls)) as executor:
            futures: dict[concurrent.futures.Future[str], int] = {}
            for i, tc in enumerate(tool_calls):
                logger.info("  Dispatching (parallel): %s(%s)...", tc.name,
                            ", ".join(f"{k}={v}" for k, v in tc.arguments.items() if k != "url"))
                future = executor.submit(
                    _route_with_timeout, self.capbus, tc.name, tc.arguments
                )
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                tc = tool_calls[i]
                try:
                    result = future.result(timeout=120)
                except Exception as e:
                    logger.warning("Parallel tool '%s' failed: %s", tc.name, e)
                    result = f"Error: {e}"
                results_by_index[i] = {"tc": tc, "result": str(result), "elapsed": 0.0}
                logger.info("  %s -> %d chars (parallel)", tc.name, len(str(result)))

        return [results_by_index[i] for i in range(len(tool_calls))]

    def _checkpoint(self) -> None:
        """Save checkpoint after each turn."""
        if self._checkpoint_mgr is None:
            return
        try:
            from agenthatch_core.offload.checkpoint import Checkpoint

            spec = self.ctx._raw_spec
            session_id = (
                spec.identity.id
                if hasattr(spec, "identity") and hasattr(spec.identity, "id")
                else spec.get("identity", {}).get("id", "default")
            )
            cp = Checkpoint(
                session_id=session_id,
                turn_count=self.ctx._turn_count,
                history=list(self.ctx.history),
                summary=(
                    {"session_intent": self.ctx.summary.session_intent,
                     "current_state": self.ctx.summary.current_state,
                     "conversation_turns": self.ctx.summary.conversation_turns,
                     "key_findings": self.ctx.summary.key_findings,
                     "tool_calls_summary": self.ctx.summary.tool_calls_summary}
                    if self.ctx.summary else None
                ),
                compact_failures=self.ctx._consecutive_compact_failures,
                cb_state=self._cb_state,
                cb_failures=self._cb_failures,
            )
            self._checkpoint_mgr.save(cp)
        except Exception as e:
            logger.warning("Checkpoint save failed: %s", e)

    def stream(
        self, user_input: str
    ) -> Generator[RichToolCallEvent | str, None, str]:
        """Streaming conversation for TUI Live rendering."""
        self.ctx._turn_count += 1

        # PRE_TURN hook
        if self._hooks:
            hook_ctx = self._hooks.execute(HookPoint.PRE_TURN, {
                "user_input": user_input,
                "turn_count": self.ctx._turn_count,
            })
            user_input = hook_ctx.get("user_input", user_input)

        self.ctx.auto_compact_check(self.llm.context_window)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        # v0.9.8: PlanLayer injection for streaming path
        if self._plan_layer is not None:
            self._plan_layer.handle_turn_start()
            plan_ctx = self._plan_layer.plan_context
            if plan_ctx and messages and messages[0]["role"] == "system":
                messages[0]["content"] += "\n" + plan_ctx
            suggestion = self._plan_layer.next_suggestion
            if suggestion and messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\n" + suggestion

        tools_for_api = [
            {"type": t.type, "function": t.function}
            if hasattr(t, "type") else
            {"type": t["type"], "function": t["function"]}
            for t in tools
        ]

        full_response_text: str = ""

        if not self._cb_allow():
            yield "Service temporarily unavailable. Please wait and try again."
            return "Service temporarily unavailable."

        # ── v0.6: Autonomous task completion ──
        has_executed_tools = False
        accumulated_text = ""
        # v1.0.1 (R4-V20): Track whether any text was yielded to the
        # user during this turn.  When the LLM calls task_complete
        # after already streaming the real answer, we must NOT yield
        # the task_complete summary (which is typically a meta-summary
        # like "已回答…") — the real answer is already in front of
        # the user.  Previously the check was
        # ``yield summary if not full_response_text else ""`` but
        # ``full_response_text`` is only assigned at loop break points
        # (lines 808/814), so it was always ``""`` inside the loop
        # and the summary was always yielded — producing the trailing
        # "已回答…" meta-summary users see in KB agent responses.
        has_yielded_text = False
        tool_stats: dict[str, int] = {}
        _consecutive_text_only = 0  # v0.8.15
        while True:
            has_yielded_tool_header = False

            try:
                gen = self._call_with_retry(
                    self.llm.stream_chat_with_tools,
                    messages=messages,
                    tools=tools_for_api,
                )
                self._cb_record(True)
            except Exception as e:
                self._cb_record(False)
                logger.error("LLM stream call failed: %s", e)
                full_response_text = (
                    f"Error communicating with model provider: {e}"
                )
                break

            response = None
            while True:
                try:
                    delta = next(gen)
                except StopIteration as e:
                    response = e.value
                    break

                if delta.type == "text":
                    # v1.0.1 (R4-V18): For KB agents, suppress pre-tool-call
                    # narration.  DeepSeek often emits a brief "let me check
                    # the knowledge base…" sentence before calling retrieve,
                    # then emits the real answer after.  Both get yielded
                    # to the TUI, producing a visually disjointed response
                    # ("先检索…据典籍所载…").  When has_executed_tools is
                    # still False AND the agent is in KB mode (max_text=0),
                    # buffer the text instead of yielding — if a tool call
                    # follows, the buffered text is discarded; if no tool
                    # call ever comes, it is yielded as the final answer
                    # via accumulated_text / full_response_text below.
                    if (
                        self._max_consecutive_text_only == 0
                        and not has_executed_tools
                    ):
                        accumulated_text += delta.content or ""
                    else:
                        accumulated_text += delta.content or ""
                        yield delta.content
                        # v1.0.1 (R4-V20): Mark that we've yielded text
                        # so task_complete knows not to yield its summary.
                        has_yielded_text = True

                elif delta.type == "reasoning":
                    # v0.7.11: Reasoning content emitted as ThinkingDelta, not visible text
                    yield ThinkingDelta(content=delta.content or "")

                elif delta.type == "tool_call_start" and not has_yielded_tool_header:
                    has_yielded_tool_header = True
                    yield RichToolCallEvent(
                        phase="start",
                        tool_name=delta.tool_name or "unknown",
                    )

            if response is None:
                break

            self._record_usage(response)

            # v0.6: detect task_complete signal
            if response.tool_calls:
                tc_names = [tc.name for tc in response.tool_calls]
                if _TASK_COMPLETE_TOOL in tc_names:
                    work_tools = [tc for tc in response.tool_calls
                                  if tc.name != _TASK_COMPLETE_TOOL]
                    if not work_tools:
                        idx = tc_names.index(_TASK_COMPLETE_TOOL)
                        summary = response.tool_calls[idx].arguments.get(
                            "summary", "Done."
                        )
                        # v1.0.1 (R4-V20): If we already streamed the real
                        # answer to the user (has_yielded_text=True), use
                        # accumulated_text as the final text — do NOT yield
                        # the task_complete summary, which is typically a
                        # meta-summary like "已回答…".  Only yield the
                        # summary when no text was streamed (e.g. the LLM
                        # called task_complete without any prior answer).
                        if has_yielded_text:
                            final_text = accumulated_text
                        else:
                            final_text = summary
                            yield summary
                        # v1.0.1 (R4-V21): Strip trailing meta-narration
                        # for KB agents.  The LLM often appends sentences
                        # like "已完整解答…无剩余步骤" before calling
                        # task_complete, despite B4 (e) forbidding it.
                        if self._max_consecutive_text_only == 0:
                            final_text = _strip_trailing_meta_narration(final_text)
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", final_text)
                        # v0.7.11: Record turn to memory
                        if self._memory_brick:
                            self._memory_brick.record_turn("user", user_input)
                            self._memory_brick.record_turn("assistant", final_text)
                        if self._hooks:
                            _ = self._hooks.execute(HookPoint.POST_TURN, {
                                "turn_count": self.ctx._turn_count,
                                "final_text": final_text,
                            })
                        self._checkpoint()
                        return cast("str", final_text)
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.tool_calls:
                # v0.6: Auto-continuation only after tools executed
                # (needsFollowUp pattern)
                if not has_executed_tools:
                    full_response_text = response.text or accumulated_text
                    break
                # v0.8.14: Limit consecutive text-only
                # v1.0.1 (R4-V17): Use instance vars (per-agent tuning).
                _consecutive_text_only += 1
                if _consecutive_text_only > self._max_consecutive_text_only:
                    full_response_text = response.text or accumulated_text
                    break
                # v0.9.8: Grace period — allow the agent to report
                # progress naturally (e.g. "Page opened, search box visible")
                # before nudging it to continue or complete.
                # This matches the run() method's grace period logic.
                if _consecutive_text_only < self._nudge_grace:
                    accumulated_text = ""
                    continue
                messages.append({
                    "role": "assistant",
                    "content": response.text or accumulated_text,
                })
                self.ctx.add_to_history("assistant", response.text or accumulated_text)
                messages.append({"role": "user", "content": _CONTINUE_NUDGE})

                # v0.9: Check interrupt before auto-continuation (streaming)
                if self._check_interrupted():
                    self.ctx.add_to_history("user", user_input)
                    return _INTERRUPT_MESSAGE

                if not self._cb_allow():
                    break
                continue

            assistant_tool_calls = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for tc in response.tool_calls
            ]
            stream_assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            }
            # v0.9.8: Preserve reasoning_content for DeepSeek thinking mode
            if response.reasoning_content:
                stream_assistant_msg["reasoning_content"] = response.reasoning_content
            messages.append(stream_assistant_msg)
            self.ctx.add_to_history(
                "assistant", None, tool_calls=assistant_tool_calls
            )

            # PRE_TOOL_CALL hook (streaming)
            if self._hooks and response.tool_calls:
                _ = self._hooks.execute(HookPoint.PRE_TOOL_CALL, {
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                })

            # v0.9: Check interrupt before executing tools (streaming)
            if self._check_interrupted():
                self.ctx.add_to_history("user", user_input)
                return _INTERRUPT_MESSAGE

            parallel = (
                self.llm.features.supports_parallel_tool_calls
                and len(response.tool_calls) > 1
            )
            # v0.9.8: Collect tool results for PlanLayer
            stream_plan_results: list[dict[str, Any]] = []
            for entry in self._execute_tool_calls(
                response.tool_calls, parallel=parallel
            ):
                tc = entry["tc"]
                elapsed = entry["elapsed"]
                result = entry["result"]
                tool_stats[tc.name] = tool_stats.get(tc.name, 0) + 1

                yield RichToolCallEvent(
                    phase="done",
                    tool_name=tc.name,
                    elapsed=elapsed,
                    result_preview=result[:200],
                )

                result_str = result
                if len(result_str) > MAX_TOOL_RESULT_CHARS:
                    result_str = result_str[:MAX_TOOL_RESULT_CHARS] + "\n... (truncated)"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })
                self.ctx.add_to_history(
                    "tool",
                    f"[{tc.name}]: {str(result)[:500]}",
                    tool_call_id=tc.id,
                )

                # POST_TOOL_CALL hook (streaming)
                if self._hooks:
                    _ = self._hooks.execute(HookPoint.POST_TOOL_CALL, {
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_str,
                        "elapsed": elapsed,
                    })
                has_executed_tools = True
                _consecutive_text_only = 0  # v0.8.14: reset on tool execution
                # v1.0.1 (R4-V18): Discard any pre-tool-call narration
                # buffered for KB agents — only the post-retrieve answer
                # should reach the user.
                if self._max_consecutive_text_only == 0:
                    accumulated_text = ""

                # v0.9.8: Collect for PlanLayer
                stream_plan_results.append({
                    "name": tc.name,
                    "content": result_str,
                })

            # v0.9.8: PlanLayer — update state after stream tool execution
            if self._plan_layer is not None and stream_plan_results:
                self._plan_layer.handle_turn_end(
                    response_text=None,
                    tool_results=stream_plan_results,
                )

        self.ctx.add_to_history("user", user_input)
        final_text = full_response_text or accumulated_text
        # v1.0.1 (R4-V21): Strip trailing meta-narration for KB agents
        # on the non-task_complete exit path too.
        if self._max_consecutive_text_only == 0:
            final_text = _strip_trailing_meta_narration(final_text)
        if final_text:
            self.ctx.add_to_history("assistant", final_text)
        # v0.7.11: Record turn to memory
        if self._memory_brick:
            self._memory_brick.record_turn("user", user_input)
            if final_text:
                self._memory_brick.record_turn("assistant", final_text)

        # POST_TURN hook (streaming)
        if self._hooks:
            _ = self._hooks.execute(HookPoint.POST_TURN, {
                "turn_count": self.ctx._turn_count,
                "final_text": final_text,
            })

        self._checkpoint()
        return final_text or "(no response)"