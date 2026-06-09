"""MemoryBrick — persistent agent memory (v0.7.6).

Provides cross-session knowledge persistence, user preference learning,
and semantic recall. File-based primary storage with optional SQLite FTS5
index and vector embeddings.

Default-on, opt-out via ``memory: false`` in brick_manifest.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .search import MemorySearch
from .store import MemoryStore
from .tools import recall_tool


class MemoryBrick:
    """Persistent agent memory brick.

    File-based primary storage (human-readable Markdown + JSONL) with
    optional SQLite FTS5 index and vector embeddings for hybrid search.

    Architecture:
        .agenthatch/memory/{skill_id}/
            MEMORY.md              ← core memory (always loaded, < 8KB)
            preferences.md         ← user preferences (evergreen)
            sessions/              ← per-session JSONL conversation logs
            index.db              ← SQLite FTS5 + optional embeddings
            knowledge/            ← extracted facts (searchable)
    """

    MAX_CORE_SIZE_BYTES = 8192  # 8KB soft limit for MEMORY.md

    def __init__(self, skill_id: str, config: dict[str, Any] | None = None):
        """Initialize memory brick for a skill.

        Args:
            skill_id: Skill identifier (e.g., "cooper").
            config: Optional config dict with keys:
                - memory_dir: Path to memory root (default: .agenthatch/memory/)
                - embedding_model: Optional embedding model name
                - embedding_api_key: API key for embedding service
        """
        config = config or {}
        self._skill_id = skill_id
        memory_root = Path(config.get("memory_dir", ".agenthatch/memory"))
        self._dir = memory_root / skill_id
        self._dir.mkdir(parents=True, exist_ok=True)

        self.store = MemoryStore(self._dir)
        self.search = MemorySearch(self.store)
        self._config = config

    # ── lifecycle ──────────────────────────────────────────────────────

    def inject_into_context(self, max_tokens: int = 1000) -> str:
        """Build memory section for system prompt injection.

        Returns core memory + preferences as a formatted string suitable
        for appending to the system prompt. Called by ContextManager.
        """
        parts: list[str] = []

        core = self.store.get_core_memory(max_tokens)
        if core:
            parts.append(core)

        prefs = self.store.get_preferences()
        if prefs:
            parts.append(f"\n## User Preferences\n{prefs}")

        return "\n".join(parts)

    def record_turn(self, role: str, content: str, tool_calls: list[dict[str, Any]] | None = None) -> None:
        """Record a conversation turn to the session log.

        Called after each user/assistant message in ConversationLoop.
        """
        self.store.append_session_entry(role, content, tool_calls)

    def save_compact_summary(self, summary: str) -> None:
        """Save a compact summary as a memory entry.

        Called after ContextManager compaction (post-compaction hook).
        """
        self.store.append_core_memory(summary)

    def remember(self, fact: str) -> None:
        """Explicitly save a fact to memory (user-triggered via /remember)."""
        self.store.save_knowledge_fact(fact)

    def recall(self, query: str, limit: int = 5) -> str:
        """Search memory for relevant entries. Exposed as the `recall` tool.

        Args:
            query: Natural language search query.
            limit: Maximum number of results (1-10).

        Returns:
            Formatted list of relevant memories with timestamps and scores.
        """
        results = self.search.search(query, top_k=min(limit, 10))
        if not results:
            return "No relevant memories found."

        lines: list[str] = []
        for r in results:
            score_str = f"(relevance: {r.score:.2f})" if r.score is not None else ""
            ts = r.timestamp or "unknown"
            lines.append(f"[{ts}] {score_str}\n{r.content}")

        return "\n\n".join(lines)

    # ── maintenance ────────────────────────────────────────────────────

    def maybe_compact_core(self) -> bool:
        """Check if core memory needs compaction (> 8KB).

        Returns True if compaction was triggered (caller should invoke LLM).
        """
        return self.store.core_size_bytes() > self.MAX_CORE_SIZE_BYTES

    def compact_core(self, compressed: str) -> None:
        """Replace core memory with LLM-compressed version."""
        self.store.write_core_memory(compressed)

    def rebuild_index(self) -> None:
        """Rebuild the FTS5 search index (e.g., after batch import)."""
        self.search.rebuild_index()