"""ConversationLoop — LLM <-> Tool calling cycle (v0.4)."""

from __future__ import annotations

import json
import logging
import random
import time
from collections.abc import Callable, Generator
from typing import Any

from agenthatch.base.sandbox import Sandbox
from agenthatch.cap.bus import CapBus
from agenthatch.exceptions import CapabilityNotFoundError
from agenthatch.skill.llm_client import LLMClient, ToolCallResponse

logger = logging.getLogger(__name__)


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

        # ── DD-05-16: Circuit breaker guard ──
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

        for _ in range(self.MAX_TOOL_ROUNDS):
            if not response.has_tool_calls:
                break

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
                try:
                    result = self.capbus.route(tc.name, tc.arguments)
                except CapabilityNotFoundError as e:
                    logger.warning("Tool call failed: %s", e)
                    result = f"Error: {e}"
                except Exception as e:
                    logger.warning("Tool execution failed: %s", e)
                    result = f"Error: {e}"
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": str(result),
                })
                self.ctx.add_to_history(
                    "tool",
                    f"[{tc.name}]: {str(result)[:500]}",
                    tool_call_id=tc.id,
                )

            # ── DD-05-16: Circuit breaker for inner LLM call ──
            if not self._cb_allow():
                response = ToolCallResponse(
                    text="Service temporarily unavailable. Please wait and try again.",
                    tool_calls=[],
                )
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

    def _commit_tool_round(
        self,
        messages: list[dict[str, Any]],
        tool_calls: list[Any],
        route_results: list[tuple[str, str, str]],
    ) -> None:
        """Commit a complete tool round to messages and history."""
        assistant_tool_calls = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in tool_calls
        ]
        messages.append({
            "role": "assistant",
            "content": None,
            "tool_calls": assistant_tool_calls,
        })
        self.ctx.add_to_history(
            "assistant", None, tool_calls=assistant_tool_calls
        )

        for (tc_id, tool_name, result) in route_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": result,
            })
            self.ctx.add_to_history(
                "tool",
                f"[{tool_name}]: {str(result)[:500]}",
                tool_call_id=tc_id,
            )

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
        self.ctx.auto_compact_check(self.llm.model_max_tokens or 4096)

        messages = self.ctx.build_messages(user_input)
        tools = self.capbus.list_tool_definitions()

        full_response_text: str = ""

        # ── DD-05-16: Circuit breaker guard ──
        if not self._cb_allow():
            yield "Service temporarily unavailable. Please wait and try again."
            return "Service temporarily unavailable."

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

            if not response.has_tool_calls:
                full_response_text = response.text or accumulated_text
                break

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
                    result = self.capbus.route(tc.name, tc.arguments)
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
                    result_preview=result[:200],
                )

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })
                self.ctx.add_to_history(
                    "tool",
                    f"[{tc.name}]: {str(result)[:500]}",
                    tool_call_id=tc.id,
                )

        self.ctx.add_to_history("user", user_input)
        final_text = full_response_text or accumulated_text
        if final_text:
            self.ctx.add_to_history("assistant", final_text)

        self._checkpoint()
        return final_text or "(no response)"
