"""MemoryStore — file-based primary storage for MemoryBrick (v0.7.6).

Human-readable Markdown (core memory, preferences) + JSONL (session logs,
knowledge facts). SQLite FTS5 index is managed by MemorySearch.
"""

from __future__ import annotations

import json
import sqlite3
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
            try:
                for line in session_file.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        entries.append(json.loads(line))
            except (json.JSONDecodeError, OSError):
                continue
        return entries

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
        entries: list[dict[str, Any]] = []
        try:
            for line in self._facts_path.read_text(encoding="utf-8").strip().split("\n"):
                if line.strip():
                    entries.append(json.loads(line))
        except (json.JSONDecodeError, OSError):
            pass
        return entries

    # ── SQLite index support ───────────────────────────────────────────

    def get_db_path(self) -> Path:
        """Return the path to the SQLite index database."""
        return self._dir / "index.db"

    def get_db(self) -> sqlite3.Connection:
        """Get or create the SQLite database connection."""
        db = sqlite3.connect(str(self.get_db_path()))
        db.execute("PRAGMA journal_mode=WAL")
        return db