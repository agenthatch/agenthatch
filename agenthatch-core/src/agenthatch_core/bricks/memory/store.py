"""MemoryStore — file-based primary storage for MemoryBrick (v0.7.6).

Human-readable Markdown (core memory, preferences) + JSONL (session logs,
knowledge facts). SQLite FTS5 index is managed by MemorySearch.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class MemoryStore:
    """File-based storage with SQLite FTS5 index support.

    Directory layout:
        {dir}/
            MEMORY.md          ← core memory (always loaded at session start)
            preferences.md     ← user preferences (evergreen, no time decay)
            sessions/          ← per-session JSONL conversation logs
                YYYY-MM-DD.jsonl
            knowledge/         ← extracted facts as JSONL
                facts.jsonl
    """

    def __init__(self, memory_dir: Path):
        self._dir = memory_dir
        self._dir.mkdir(parents=True, exist_ok=True)
        self._sessions_dir = self._dir / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)
        self._knowledge_dir = self._dir / "knowledge"
        self._knowledge_dir.mkdir(parents=True, exist_ok=True)
        self._core_path = self._dir / "MEMORY.md"
        self._prefs_path = self._dir / "preferences.md"
        self._facts_path = self._knowledge_dir / "facts.jsonl"
        # v1.0.9 (Bug 19): Thread-local SQLite connection cache.
        # Previously get_db() opened a brand-new connection on every call
        # and no caller closed it. A long-running agent calling recall()
        # once per turn for 1000 turns would leak ~2000 FDs (each sqlite
        # connection holds 2 FDs: the db file + WAL) and hit the OS fd
        # limit. Per project_memory.md hard constraint: "SQLite connections
        # must use thread-local storage to prevent cross-thread errors".
        self._thread_local = threading.local()

    # ── core memory ────────────────────────────────────────────────────

    def get_core_memory(self, max_tokens: int = 1000) -> str:
        """Read core memory, truncated to max_tokens (chars / 4)."""
        if not self._core_path.exists():
            return ""
        content = self._core_path.read_text(encoding="utf-8")
        max_chars = max_tokens * 4
        if len(content) > max_chars:
            content = content[:max_chars] + "\n\n... (truncated)"
        return content

    def append_core_memory(self, entry: str) -> None:
        """Append an entry to core memory."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        line = f"\n\n## [{timestamp}]\n{entry}"
        with open(self._core_path, "a", encoding="utf-8") as f:
            f.write(line)

    def write_core_memory(self, content: str) -> None:
        """Overwrite core memory (e.g., after LLM compaction)."""
        self._core_path.write_text(content, encoding="utf-8")

    def core_size_bytes(self) -> int:
        """Return the size of core memory in bytes."""
        if not self._core_path.exists():
            return 0
        return self._core_path.stat().st_size

    # ── preferences ────────────────────────────────────────────────────

    def get_preferences(self) -> str:
        """Read user preferences (evergreen, exempt from time decay)."""
        if not self._prefs_path.exists():
            return ""
        return self._prefs_path.read_text(encoding="utf-8").strip()

    def save_preference(self, pref: str) -> None:
        """Append a user preference."""
        with open(self._prefs_path, "a", encoding="utf-8") as f:
            f.write(f"\n- {pref}")

    # ── session logs ───────────────────────────────────────────────────

    def append_session_entry(
        self,
        role: str,
        content: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        """Append a single message to today's session log (JSONL)."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        session_path = self._sessions_dir / f"{today}.jsonl"

        entry: dict[str, Any] = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if tool_calls:
            entry["tool_calls"] = tool_calls

        with open(session_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def iter_session_entries(self) -> list[dict[str, Any]]:
        """Iterate all session entries across all session files."""
        entries: list[dict[str, Any]] = []
        for session_file in sorted(self._sessions_dir.glob("*.jsonl")):
            # v1.0.11 (Bug 29): Catch UnicodeDecodeError too — it's a
            # subclass of ValueError (via UnicodeError), NOT OSError,
            # so the previous ``except (json.JSONDecodeError, OSError)``
            # didn't catch it.  A session file with non-UTF-8 bytes
            # (e.g. truncated multi-byte sequence from a crash) would
            # propagate the error and crash the entire iter call.
            try:
                text = session_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # v1.0.11 (Bug 28): Skip bad LINES, not bad FILES.  The
            # previous code wrapped the entire read+parse loop in one
            # try/except — a single corrupt line (e.g. partial write
            # from a killed process) caused ALL valid entries in the
            # same file to be lost.  Per-line try/except preserves the
            # valid entries and only drops the bad line.
            for line in text.strip().split("\n"):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def get_recent_session_entries(self, limit: int = 30) -> list[dict[str, Any]]:
        """v0.7.13: Get the most recent session entries across all files.

        Reads session files in reverse chronological order (newest first)
        and collects entries until limit is reached. Returns in
        chronological order for context injection.
        """
        entries: list[dict[str, Any]] = []
        for session_file in sorted(
            self._sessions_dir.glob("*.jsonl"), reverse=True
        ):
            # v1.0.11 (Bug 29): Catch UnicodeDecodeError (subclass of
            # ValueError, not OSError) so a truncated-UTF-8 file skips
            # cleanly instead of crashing the iter.
            try:
                text = session_file.read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                continue
            # v1.0.11 (Bug 28): Per-line try/except — skip bad lines,
            # not the entire file.
            for line in reversed(text.strip().split("\n")):
                if not line.strip():
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
                if len(entries) >= limit:
                    break
            if len(entries) >= limit:
                break
        return list(reversed(entries))  # chronological order

    # ── knowledge facts ────────────────────────────────────────────────

    def save_knowledge_fact(self, fact: str) -> None:
        """Save an extracted fact to the knowledge store."""
        entry = {
            "fact": fact,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        with open(self._facts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def iter_knowledge_facts(self) -> list[dict[str, Any]]:
        """Iterate all knowledge facts."""
        if not self._facts_path.exists():
            return []
        # v1.0.11 (Bug 29): Catch UnicodeDecodeError (subclass of
        # ValueError, not OSError) so a truncated-UTF-8 facts file
        # returns [] instead of crashing the agent's recall.
        try:
            text = self._facts_path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return []
        entries: list[dict[str, Any]] = []
        # v1.0.11 (Bug 28): Per-line try/except — skip bad lines, not
        # the entire file.  A single corrupt line shouldn't lose all
        # valid facts in the same file.
        for line in text.strip().split("\n"):
            if not line.strip():
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return entries

    # ── SQLite index support ───────────────────────────────────────────

    def get_db_path(self) -> Path:
        """Return the path to the SQLite index database."""
        return self._dir / "index.db"

    def get_db(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite database connection.

        v1.0.9 (Bug 19): Returns a cached connection per thread instead
        of opening a new one on every call. Callers (``_ensure_index``,
        ``_bm25_search``, ``_fallback_search``, ``rebuild_index``) used
        to leak connections because none of them called ``db.close()``.
        Now the connection lives for the thread's lifetime and is reused
        across calls.
        """
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            return conn
        conn = sqlite3.connect(str(self.get_db_path()), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.OperationalError:
            pass  # WAL may fail on network filesystems — fall back to default
        self._thread_local.conn = conn
        return conn

    def close(self) -> None:
        """Close the thread-local SQLite connection if open.

        v1.0.9 (Bug 19): Optional cleanup method. MemoryBrick should call
        this on shutdown to release the connection; if it doesn't, Python's
        GC will reap it eventually, but explicit close avoids fd pressure
        on long-lived processes.
        """
        conn = getattr(self._thread_local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
            self._thread_local.conn = None