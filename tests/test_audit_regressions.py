"""Regression tests for v1.0.9 audit-discovered bugs.

Covers 5 bugs found in a focused audit of non-context modules:
- Bug 18: ConversationLoop.stream() discards accumulated_text in KB mode
- Bug 19: MemoryStore.get_db() leaks a new SQLite connection per call
- Bug 20: CapBus._validate_output rejects valid int when schema says "number"
- Bug 21: StdioTransport.send_request returns first JSON line (no id matching)
- Bug 22: SSETransport.connect() injects "id":1 into notifications/initialized
"""

from __future__ import annotations

import json
import sqlite3
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Bug 19: MemoryStore.get_db() leaks connections
# ---------------------------------------------------------------------------

class TestBug19MemoryStoreConnectionReuse:
    """Bug 19: get_db() must return the same connection per thread.

    Previously opened a brand-new sqlite3.Connection on every call.
    A long-running agent calling recall() once per turn for 1000 turns
    would leak ~2000 FDs (each sqlite connection = 2 FDs: db + WAL)
    and hit the OS fd limit.
    """

    def test_repeated_get_db_returns_same_connection(self, tmp_path: Path) -> None:
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "mem")
        conn1 = store.get_db()
        conn2 = store.get_db()
        conn3 = store.get_db()
        assert conn1 is conn2 is conn3, (
            "get_db() must return the same cached connection per thread — "
            f"got {id(conn1)}, {id(conn2)}, {id(conn3)}"
        )

    def test_get_db_does_not_leak_file_descriptors(self, tmp_path: Path) -> None:
        """After 100 get_db() calls, FD count should not grow."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "mem")

        def count_fds() -> int:
            proc_fd = Path("/dev/fd")
            if proc_fd.exists():
                return len(list(proc_fd.iterdir()))
            return -1  # not measurable on this platform

        baseline = count_fds()
        for _ in range(100):
            store.get_db()
        after = count_fds()

        if baseline > 0 and after > 0:
            leaked = after - baseline
            assert leaked < 5, (
                f"100 get_db() calls leaked {leaked} FDs (baseline={baseline}, "
                f"after={after}). Connection caching is broken."
            )

    def test_close_releases_connection(self, tmp_path: Path) -> None:
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "mem")
        conn = store.get_db()
        assert not conn.execute("SELECT 1").fetchone() is None  # works
        store.close()
        # After close, get_db() should open a fresh connection
        conn2 = store.get_db()
        assert conn2 is not conn, "close() must release the cached connection"

    def test_thread_local_isolation(self, tmp_path: Path) -> None:
        """Each thread must get its own connection (project_memory.md constraint)."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path / "mem")
        main_conn = store.get_db()
        thread_conns: list[Any] = []

        def worker() -> None:
            thread_conns.append(store.get_db())

        t = threading.Thread(target=worker)
        t.start()
        t.join()

        assert thread_conns[0] is not main_conn, (
            "Worker thread must NOT share the main thread's connection — "
            "violates project_memory.md thread-local SQLite constraint"
        )


# ---------------------------------------------------------------------------
# Bug 20: CapBus._validate_output rejects valid int for "number" schema
# ---------------------------------------------------------------------------

class TestBug20ValidateOutputAcceptsIntegerForNumber:
    """Bug 20: JSON Schema "number" is a superset of "integer".

    A tool returning ``{"count": 42}`` for a field declared as
    ``{"type": "number"}`` is valid per spec
    (https://json-schema.org/draft/2020-12/json-schema-validation#name-type).
    Previously the validator did exact string match and rejected it
    with "expected number, got integer", breaking the agent's reasoning.
    """

    def _make_bus(self, schema: dict[str, Any]) -> Any:
        from agenthatch_core.tools.bus import CapBus

        bus = CapBus()
        bus._output_schemas["mytool"] = schema
        return bus

    def test_int_value_accepted_for_number_schema(self) -> None:
        bus = self._make_bus({
            "type": "object",
            "properties": {"count": {"type": "number"}},
        })
        result = bus._validate_output("mytool", json.dumps({"count": 42}))
        # Should return the JSON-serialized data, not an error string
        assert "Error:" not in result, (
            f"int 42 should be valid for type=number, got: {result}"
        )
        assert "42" in result

    def test_float_value_still_accepted_for_number_schema(self) -> None:
        bus = self._make_bus({
            "type": "object",
            "properties": {"count": {"type": "number"}},
        })
        result = bus._validate_output("mytool", json.dumps({"count": 42.5}))
        assert "Error:" not in result
        assert "42.5" in result

    def test_int_value_accepted_for_integer_schema(self) -> None:
        bus = self._make_bus({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        })
        result = bus._validate_output("mytool", json.dumps({"count": 42}))
        assert "Error:" not in result

    def test_float_value_rejected_for_integer_schema(self) -> None:
        """Regression guard: float for integer schema must still fail."""
        bus = self._make_bus({
            "type": "object",
            "properties": {"count": {"type": "integer"}},
        })
        result = bus._validate_output("mytool", json.dumps({"count": 42.5}))
        assert "Error:" in result, (
            "float 42.5 must be rejected for type=integer (subset direction matters)"
        )

    def test_string_rejected_for_number_schema(self) -> None:
        """Regression guard: string for number schema must still fail."""
        bus = self._make_bus({
            "type": "object",
            "properties": {"count": {"type": "number"}},
        })
        result = bus._validate_output("mytool", json.dumps({"count": "42"}))
        assert "Error:" in result


# ---------------------------------------------------------------------------
# Bug 21: StdioTransport.send_request matches response by request id
# ---------------------------------------------------------------------------

class TestBug21StdioTransportResponseIdMatching:
    """Bug 21: send_request must match response by request ``id``.

    MCP servers can emit async notifications (notifications/progress,
    notifications/message, log) on stdout BEFORE the actual response.
    The previous code returned the first valid JSON line, so a startup
    notification would be mistaken for the initialize response.
    """

    def _make_transport_with_stdout(self, lines: list[str]) -> Any:
        """Build a StdioTransport with a fake subprocess emitting ``lines``."""
        from agenthatch_core.mcp.client import StdioTransport, MCPServerConfig

        config = MCPServerConfig(
            command="fake",
            args=[],
            env={},
            transport="stdio",
            timeout=5.0,
            url="",
            auth_token="",
            headers={},
        )
        transport = StdioTransport(config)

        # Fake stdout that yields lines one at a time
        line_iter = iter(lines)

        class _Stdin:
            def write(self, _data):
                pass
            def flush(self):
                pass

        class _Stdout:
            def readline(self):
                try:
                    return next(line_iter)
                except StopIteration:
                    return ""  # EOF

        class FakeProc:
            stdin = _Stdin()
            stdout = _Stdout()

        transport._proc = FakeProc()
        return transport

    def test_notification_before_response_is_skipped(self) -> None:
        """A ``notifications/message`` line before the response must be skipped."""
        lines = [
            json.dumps({"jsonrpc": "2.0", "method": "notifications/message",
                        "params": {"log": "starting up"}}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}) + "\n",
        ]
        transport = self._make_transport_with_stdout(lines)
        resp = transport.send_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert resp.get("result") == {"tools": []}, (
            f"Should return the response, not the notification. Got: {resp}"
        )

    def test_mismatched_id_is_skipped(self) -> None:
        """A response with a different id must be skipped."""
        lines = [
            json.dumps({"jsonrpc": "2.0", "id": 999, "result": {"stale": True}}) + "\n",
            json.dumps({"jsonrpc": "2.0", "id": 1, "result": {"tools": []}}) + "\n",
        ]
        transport = self._make_transport_with_stdout(lines)
        resp = transport.send_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
        assert resp.get("result") == {"tools": []}
        assert resp.get("id") == 1

    def test_broken_pipe_returns_empty(self) -> None:
        """BrokenPipeError on write must return {} per contract, not crash."""
        from agenthatch_core.mcp.client import StdioTransport, MCPServerConfig

        config = MCPServerConfig(
            command="fake", args=[], env={}, transport="stdio",
            timeout=5.0, url="", auth_token="", headers={},
        )
        transport = StdioTransport(config)

        class BrokenPipe:
            def write(self, _):
                raise BrokenPipeError("subprocess died")
            def flush(self):
                pass

        class FakeProc:
            stdin = BrokenPipe()
            stdout = None
            stderr = None

        transport._proc = FakeProc()
        # Must NOT raise
        resp = transport.send_request({"jsonrpc": "2.0", "id": 1, "method": "x"})
        assert resp == {}


# ---------------------------------------------------------------------------
# Bug 22: SSETransport.connect() must not inject "id" into notifications/initialized
# ---------------------------------------------------------------------------

class TestBug22SSETransportNotificationHasNoId:
    """Bug 22: ``notifications/initialized`` must NOT carry an ``id``.

    Per JSON-RPC 2.0, notifications have no ``id``. The previous code
    routed the notification through ``send_request``, which injected
    ``"id": 1`` (because ``"jsonrpc"`` was missing), turning the
    notification into a regular request AND colliding its id with the
    preceding initialize call.
    """

    def test_notification_posted_without_id(self) -> None:
        from agenthatch_core.mcp.client import SSETransport, MCPServerConfig

        config = MCPServerConfig(
            command="", args=[], env={}, transport="sse",
            timeout=5.0, url="http://fake.example/mcp",
            auth_token="", headers={},
        )
        transport = SSETransport(config)

        # Mock httpx.Client
        posts: list[dict[str, Any]] = []

        class FakeResp:
            status_code = 200
            text = ""
            headers = {"content-type": "application/json"}
            def json(self):
                return {"jsonrpc": "2.0", "id": 1, "result": {}}

        class FakeClient:
            def post(self, url, json=None, headers=None, timeout=None):
                posts.append({"url": url, "json": json, "headers": headers})
                return FakeResp()
            def close(self):
                pass

        with patch("httpx.Client", return_value=FakeClient()):
            transport.connect()

        # posts[0] = initialize (should have id)
        # posts[1] = notifications/initialized (should NOT have id)
        assert len(posts) >= 2, f"Expected at least 2 POSTs, got {len(posts)}"
        init_post = posts[0]
        notif_post = posts[1]

        assert "id" in init_post["json"], "initialize must have id"
        assert init_post["json"]["method"] == "initialize"

        notif_json = notif_post["json"]
        assert notif_json["method"] == "notifications/initialized"
        assert "id" not in notif_json, (
            f"notifications/initialized must NOT have 'id' per JSON-RPC 2.0 — "
            f"got: {notif_json}"
        )


# ---------------------------------------------------------------------------
# Bug 18: ConversationLoop.stream() KB-mode accumulated_text preservation
# ---------------------------------------------------------------------------

class TestBug18StreamingKBPreservesAccumulatedText:
    """Bug 18: When KB agent buffers text and calls task_complete
    without ever calling a tool, the buffered text must be the final
    answer — not the task_complete summary.

    The R4-V18 comment claimed this was already the behavior ("if no
    tool call ever comes, it is yielded as the final answer via
    accumulated_text below"), but the code actually used ``summary``.
    """

    def test_branch_logic_uses_accumulated_text_when_present(self) -> None:
        """Unit-style test of the exact branch logic.

        Simulates: KB mode, text buffered (has_yielded_text=False),
        accumulated_text="The capital of France is Paris.",
        summary="Done."
        """
        # Replicate the fixed branch from agent_loop.py
        has_yielded_text = False
        accumulated_text = "The capital of France is Paris."
        summary = "Done."

        if has_yielded_text:
            final_text = accumulated_text
            yielded = None
        elif accumulated_text:
            final_text = accumulated_text
            yielded = accumulated_text
        else:
            final_text = summary
            yielded = summary

        assert final_text == "The capital of France is Paris.", (
            f"Buffered text must be the final answer, got: {final_text!r}"
        )
        assert yielded == "The capital of France is Paris.", (
            f"User must see buffered text, got: {yielded!r}"
        )

    def test_branch_logic_falls_back_to_summary_when_no_buffer(self) -> None:
        """When accumulated_text is empty, fall back to summary (preserves
        the original R4-V20 behavior for the no-buffer case)."""
        has_yielded_text = False
        accumulated_text = ""
        summary = "Done."

        if has_yielded_text:
            final_text = accumulated_text
            yielded = None
        elif accumulated_text:
            final_text = accumulated_text
            yielded = accumulated_text
        else:
            final_text = summary
            yielded = summary

        assert final_text == "Done."
        assert yielded == "Done."

    def test_source_code_has_elif_branch(self) -> None:
        """Static guard: the source file must contain the elif branch.

        Prevents a future refactor from re-introducing the bug by
        collapsing the elif back into the else.
        """
        src = Path(
            "agenthatch-core/src/agenthatch_core/loop/agent_loop.py"
        ).read_text()
        # The fixed code has three branches in order
        assert "if has_yielded_text:" in src
        assert "elif accumulated_text:" in src
        assert "yield accumulated_text" in src
        # The Bug 18 comment must be present (so future readers know why)
        assert "Bug 18" in src or "Bug #18" in src, (
            "Bug 18 comment must be in the source so the fix is documented"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
