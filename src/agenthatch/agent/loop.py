"""ConversationLoop — LLM <-> Tool calling cycle (v0.4)."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import time
from collections.abc import Callable, Generator
from typing import Any, cast

from agenthatch.agent.hooks import HookPoint
from agenthatch.base.sandbox import Sandbox
from agenthatch.cap.bus import CapBus
from agenthatch.exceptions import CapabilityNotFoundError
from agenthatch.skill.llm_client import LLMClient, ToolCallResponse

logger = logging.getLogger(__name__)

MAX_TOOL_RESULT_CHARS = 10000

# ── v0.6 Autonomous task completion ──────────────────────────────────
_TASK_COMPLETE_TOOL = "task_complete"
_CONTINUE_NUDGE = (
    "Task not complete. Continue working on the user's request. "
    "If all steps are done, call task_complete with a summary."
)


def _route_with_timeout(
    capbus: CapBus, tool_name: str, arguments: dict[str, Any], timeout: int = 120
) -> str:
    """Execute tool with timeout to prevent infinite hangs."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(capbus.route, tool_name, arguments)
        try:
            return future.result(timeout=timeout)
        except concurrent.futures.TimeoutError:
            return f"Error: tool '{tool_name}' timed out after {timeout}s"


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


class ConversationLoop:
    """Drives the User -> LLM -> Tool -> LLM -> Response cycle."""

    MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        llm: LLMClient,
        capbus: CapBus,
        sandbox: Sandbox,
        ctx: Any,
        hooks: Any = None,
    ):
        self.llm = llm
        self.capbus = capbus
        self.sandbox = sandbox
        self.ctx = ctx
        self._hooks = hooks

        self._max_retries: int = 1
        self._retry_base_delay: float = 1.0
        self._retry_max_delay: float = 30.0
        self._retryable_statuses: set[int] = {429, 500, 502, 503, 504}

        self._cb_threshold: int = 5
        self._cb_timeout: float = 60.0
        self._cb_failures: int = 0
        self._cb_state: str = "closed"
        self._cb_opened_at: float = 0.0

        self._checkpoint_mgr: Any = None

    def run(self, user_input: str) -> str:
        """Execute one conversation turn synchronously."""
        self.ctx._turn_count += 1

        # PRE_TURN hook
        if self._hooks:
            turn_ctx: dict[str, Any] = {
                "user_input": user_input,
                "turn_count": self.ctx._turn_count,
            }
            self._hooks.execute(HookPoint.PRE_TURN, turn_ctx)

        self.ctx.auto_compact_check(self.llm.context_window)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        # Circuit breaker guard
        if not self._cb_allow():
            return "Service temporarily unavailable. Please wait and try again."

        try:
            response = self._call_with_retry(
                self.llm.chat_with_tools, messages=messages, tools=tools,
            )
            self._cb_record(True)
        except Exception as e:
            self._cb_record(False)
            logger.error("LLM API call failed: %s", e)
            return f"I encountered an error communicating with the model provider: {e}"

        task_completed = False
        has_executed_tools = False
        for _ in range(self.MAX_TOOL_ROUNDS):
            # v0.6: detect task_complete signal, return summary
            if response.has_tool_calls:
                tc_names = [tc.name for tc in response.tool_calls]
                if _TASK_COMPLETE_TOOL in tc_names:
                    work_tools = [tc for tc in response.tool_calls
                                  if tc.name != _TASK_COMPLETE_TOOL]
                    if not work_tools:
                        summary = response.tool_calls[
                            tc_names.index(_TASK_COMPLETE_TOOL)
                        ].arguments.get("summary", "Done.")
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", summary)
                        if self._hooks:
                            self._hooks.execute(HookPoint.POST_TURN, {
                                "turn_count": self.ctx._turn_count,
                                "final_text": summary,
                            })
                        self._checkpoint()
                        return cast("str", summary)
                    # Mixed: task_complete + work tools — execute work, defer completion
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.has_tool_calls:
                # v0.6: Auto-continuation only when tools were previously
                # executed (needsFollowUp pattern).
                # First-response text-only = terminal.
                if not has_executed_tools:
                    break
                messages.append({
                    "role": "assistant",
                    "content": response.text or "",
                })
                self.ctx.add_to_history("assistant", response.text)
                messages.append({"role": "user", "content": _CONTINUE_NUDGE})

                if not self._cb_allow():
                    break
                try:
                    response = self._call_with_retry(
                        self.llm.chat_with_tools, messages, tools,
                    )
                    self._cb_record(True)
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
            messages.append(assistant_msg)

            self.ctx.add_to_history(
                "assistant",
                None,
                tool_calls=assistant_msg.get("tool_calls"),
            )

            # PRE_TOOL_CALL hook
            if self._hooks and response.tool_calls:
                self._hooks.execute(HookPoint.PRE_TOOL_CALL, {
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                })

            parallel = (
                self.llm.features.supports_parallel_tool_calls
                and len(response.tool_calls) > 1
            )
            for entry in self._execute_tool_calls(
                response.tool_calls, parallel=parallel
            ):
                tc = entry["tc"]
                elapsed = entry["elapsed"]
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
                    self._hooks.execute(HookPoint.POST_TOOL_CALL, {
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_str,
                        "elapsed": elapsed,
                    })
                has_executed_tools = True

            # Circuit breaker for inner LLM call
            if not self._cb_allow():
                break

            try:
                response = self._call_with_retry(
                    self.llm.chat_with_tools, messages, tools,
                )
                self._cb_record(True)
            except Exception as e:
                self._cb_record(False)
                logger.error("LLM API call failed in tool loop: %s", e)
                response = ToolCallResponse(
                    text=f"Error communicating with model provider: {e}",
                    tool_calls=[],
                )
        else:
            # v0.6: Max rounds exhausted — synthesize best-effort summary
            # (_handle_max_steps_reached pattern)
            task_completed = True

        # ── v0.6: Max-rounds fallback ──
        if task_completed:
            self.ctx.add_to_history("user", user_input)
            messages.append({
                "role": "user",
                "content": "Summarize what you accomplished in 1-3 sentences.",
            })
            try:
                response = self._call_with_retry(
                    self.llm.chat_with_tools, messages, tools,
                )
                self._cb_record(True)
            except Exception as e:
                self._cb_record(False)
                logger.error("Fallback summarization failed: %s", e)
                response = ToolCallResponse(
                    text="(Task partially completed — max rounds reached)",
                    tool_calls=[],
                )

        self.ctx.add_to_history("user", user_input)
        final_text = response.text if response and response.text else ""
        if final_text:
            self.ctx.add_to_history("assistant", final_text)

        # POST_TURN hook
        if self._hooks:
            self._hooks.execute(HookPoint.POST_TURN, {
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

        # Parallel path: submit all, collect with as_completed, sort back
        results_by_index: dict[int, dict[str, Any]] = {}
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures: dict[concurrent.futures.Future[str], int] = {}
            for i, tc in enumerate(tool_calls):
                logger.info("  Dispatching (parallel): %s(%s)...", tc.name,
                            ", ".join(f"{k}={v}" for k, v in tc.arguments.items() if k != "url"))
                future = executor.submit(
                    self.capbus.route, tc.name, tc.arguments
                )
                futures[future] = i

            for future in concurrent.futures.as_completed(futures):
                i = futures[future]
                tc = tool_calls[i]
                try:
                    result = str(future.result(timeout=120))
                except (CapabilityNotFoundError, Exception) as e:
                    logger.warning("Parallel tool '%s' failed: %s", tc.name, e)
                    result = f"Error: {e}"
                results_by_index[i] = {"tc": tc, "result": result, "elapsed": 0.0}
                logger.info("  %s -> %d chars (parallel)", tc.name, len(result))

        return [results_by_index[i] for i in range(len(tool_calls))]

    def _checkpoint(self) -> None:
        """Save checkpoint after each turn."""
        if self._checkpoint_mgr is None:
            return
        try:
            from agenthatch.agent.offload import Checkpoint

            cp = Checkpoint(
                session_id=self.ctx.spec.identity.id,
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
            self._hooks.execute(HookPoint.PRE_TURN, {
                "user_input": user_input,
                "turn_count": self.ctx._turn_count,
            })

        self.ctx.auto_compact_check(self.llm.context_window)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        full_response_text: str = ""

        # Circuit breaker guard
        if not self._cb_allow():
            yield "Service temporarily unavailable. Please wait and try again."
            return "Service temporarily unavailable."

        # ── v0.6: Autonomous task completion ──
        task_completed = False
        has_executed_tools = False
        for _ in range(self.MAX_TOOL_ROUNDS):
            accumulated_text = ""
            has_yielded_tool_header = False

            try:
                gen = self._call_with_retry(
                    self.llm.stream_chat_with_tools,
                    messages=messages,
                    tools=tools,
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
                    accumulated_text += delta.content
                    yield delta.content

                elif delta.type == "tool_call_start" and not has_yielded_tool_header:
                    has_yielded_tool_header = True
                    yield RichToolCallEvent(
                        phase="start",
                        tool_name=delta.tool_name or "unknown",
                    )

            if response is None:
                break

            # v0.6: detect task_complete signal
            if response.has_tool_calls:
                tc_names = [tc.name for tc in response.tool_calls]
                if _TASK_COMPLETE_TOOL in tc_names:
                    work_tools = [tc for tc in response.tool_calls
                                  if tc.name != _TASK_COMPLETE_TOOL]
                    if not work_tools:
                        idx = tc_names.index(_TASK_COMPLETE_TOOL)
                        summary = response.tool_calls[idx].arguments.get(
                            "summary", "Done."
                        )
                        yield summary if not full_response_text else ""
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", summary)
                        if self._hooks:
                            self._hooks.execute(HookPoint.POST_TURN, {
                                "turn_count": self.ctx._turn_count,
                                "final_text": summary,
                            })
                        self._checkpoint()
                        return cast("str", summary)
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.has_tool_calls:
                # v0.6: Auto-continuation only after tools executed
                # (needsFollowUp pattern)
                if not has_executed_tools:
                    full_response_text = response.text or accumulated_text
                    break
                messages.append({
                    "role": "assistant",
                    "content": response.text or accumulated_text,
                })
                self.ctx.add_to_history("assistant", response.text or accumulated_text)
                messages.append({"role": "user", "content": _CONTINUE_NUDGE})

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
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": assistant_tool_calls,
            })
            self.ctx.add_to_history(
                "assistant", None, tool_calls=assistant_tool_calls
            )

            # PRE_TOOL_CALL hook (streaming)
            if self._hooks and response.tool_calls:
                self._hooks.execute(HookPoint.PRE_TOOL_CALL, {
                    "tool_calls": [
                        {"name": tc.name, "arguments": tc.arguments}
                        for tc in response.tool_calls
                    ],
                })

            parallel = (
                self.llm.features.supports_parallel_tool_calls
                and len(response.tool_calls) > 1
            )
            for entry in self._execute_tool_calls(
                response.tool_calls, parallel=parallel
            ):
                tc = entry["tc"]
                elapsed = entry["elapsed"]
                result = entry["result"]

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
                    self._hooks.execute(HookPoint.POST_TOOL_CALL, {
                        "tool_name": tc.name,
                        "arguments": tc.arguments,
                        "result": result_str,
                        "elapsed": elapsed,
                    })
                has_executed_tools = True
        else:
            task_completed = True

        # ── v0.6: Max-rounds fallback ──
        if task_completed:
            yield "(Max rounds reached, summarizing...)"
            messages.append({
                "role": "user",
                "content": "Summarize what you accomplished in 1-3 sentences.",
            })
            try:
                response = self._call_with_retry(
                    self.llm.chat_with_tools, messages, tools,
                )
                self._cb_record(True)
                full_response_text = response.text or ""
            except Exception as e:
                self._cb_record(False)
                logger.error("Fallback summarization failed: %s", e)
                full_response_text = "(Task partially completed — max rounds reached)"

        self.ctx.add_to_history("user", user_input)
        final_text = full_response_text or accumulated_text
        if final_text:
            self.ctx.add_to_history("assistant", final_text)

        # POST_TURN hook (streaming)
        if self._hooks:
            self._hooks.execute(HookPoint.POST_TURN, {
                "turn_count": self.ctx._turn_count,
                "final_text": final_text,
            })

        self._checkpoint()
        return final_text or "(no response)"
