"""ConversationLoop — LLM <-> Tool calling cycle (agenthatch-core)."""

from __future__ import annotations

import concurrent.futures
import json
import logging
import random
import time
from collections.abc import Callable, Generator
from typing import Any

from agenthatch_core.exceptions import CapabilityNotFoundError
from agenthatch_core.llm.client import LLMClient
from agenthatch_core.llm.types import ToolCallResponse
from agenthatch_core.sandbox.executor import Sandbox
from agenthatch_core.tools.bus import CapBus

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


class ConversationLoop:
    """Drives the User -> LLM -> Tool -> LLM -> Response cycle."""

    MAX_TOOL_ROUNDS = 10

    def __init__(
        self,
        llm: LLMClient,
        capbus: CapBus,
        sandbox: Sandbox,
        ctx: Any,
    ):
        self.llm = llm
        self.capbus = capbus
        self.sandbox = sandbox
        self.ctx = ctx

        self._max_retries: int = 3
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
        self.ctx.auto_compact_check(self.llm.model_max_tokens or 4096)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()
        tools_for_api = [{"type": t.type, "function": t.function} for t in tools]

        if not self._cb_allow():
            return "Service temporarily unavailable. Please wait and try again."

        # v0.6: first round forces tool choice (Instructor pattern — LLM must act)
        try:
            response = self._call_with_retry(
                self.llm.chat_with_tools, messages=messages, tools=tools_for_api,
                tool_choice="required",
            )
            self._cb_record(True)
        except Exception as e:
            self._cb_record(False)
            logger.error("LLM API call failed: %s", e)
            return f"I encountered an error communicating with the model provider: {e}"

        task_completed = False
        for _ in range(self.MAX_TOOL_ROUNDS):
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
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", summary)
                        self._checkpoint()
                        return summary
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.tool_calls:
                # v0.6: Auto-continuation — text-only is a status update
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
                        self.llm.chat_with_tools, messages, tools_for_api,
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

            for tc in response.tool_calls:
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
                except CapabilityNotFoundError as e:
                    result = f"Error: {e}"
                except Exception as e:
                    result = f"Error: {e}"
                elapsed = time.time() - t0
                logger.info(
                    "  %s -> %d chars (%.1fs)", tc.name,
                    len(str(result)), elapsed,
                )
                result_str = str(result)
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

            if not self._cb_allow():
                break

            try:
                response = self._call_with_retry(
                    self.llm.chat_with_tools, messages, tools_for_api,
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
                    self.llm.chat_with_tools, messages, tools_for_api,
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

    def _checkpoint(self) -> None:
        """Save checkpoint after each turn."""
        if self._checkpoint_mgr is None:
            return
        try:
            from agenthatch.agent.offload import Checkpoint

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
        self.ctx.auto_compact_check(self.llm.model_max_tokens or 4096)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()
        tools_for_api = [{"type": t.type, "function": t.function} for t in tools]

        full_response_text: str = ""

        if not self._cb_allow():
            yield "Service temporarily unavailable. Please wait and try again."
            return "Service temporarily unavailable."

        # ── v0.6: First round forces tool choice ──
        task_completed = False
        for round_idx in range(self.MAX_TOOL_ROUNDS):
            accumulated_text = ""
            has_yielded_tool_header = False

            try:
                gen = self._call_with_retry(
                    self.llm.stream_chat_with_tools,
                    messages=messages,
                    tools=tools_for_api,
                    tool_choice="required" if round_idx == 0 else "auto",
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
                    accumulated_text += delta.content or ""
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
                        yield summary if not full_response_text else ""
                        self.ctx.add_to_history("user", user_input)
                        self.ctx.add_to_history("assistant", summary)
                        self._checkpoint()
                        return summary
                    logger.warning(
                        "task_complete called alongside %d other tools — "
                        "executing work tools, deferring completion",
                        len(work_tools),
                    )
                    response.tool_calls = work_tools

            if not response.tool_calls:
                # v0.6: Auto-continuation — text-only is a status update
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

            for tc in response.tool_calls:
                t0 = time.time()
                try:
                    result = _route_with_timeout(self.capbus, tc.name, tc.arguments)
                except CapabilityNotFoundError as e:
                    logger.warning("Tool call failed: %s", e)
                    result = f"Error: {e}"
                except Exception as e:
                    logger.warning("Tool execution failed: %s", e)
                    result = f"Error: {e}"
                elapsed = time.time() - t0

                yield RichToolCallEvent(
                    phase="done",
                    tool_name=tc.name,
                    elapsed=elapsed,
                    result_preview=str(result)[:200],
                )

                result_str = str(result)
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
                    self.llm.chat_with_tools, messages, tools_for_api,
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

        self._checkpoint()
        return final_text or "(no response)"