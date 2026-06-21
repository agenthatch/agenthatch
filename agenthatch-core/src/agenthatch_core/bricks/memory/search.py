"""MemorySearch — hybrid search for MemoryBrick (v0.7.6).

Three-tier retrieval:
  1. BM25 via SQLite FTS5 (always available, zero dependencies)
  2. BM25 + Vector (7:3 weighted fusion) — when embedding model configured
  3. Pure keyword grep — fallback if FTS5 is unavailable

Includes time decay (30-day half-life), evergreen exemption for preferences,
and MMR reranking for diversity.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .store import MemoryStore

# Time decay: e^(-λ × days), 30-day half-life as per OpenClaw
HALF_LIFE_DAYS = 30
DECAY_LAMBDA = 0.693 / HALF_LIFE_DAYS


@dataclass
class MemoryEntry:
    """A single search result from memory."""
    content: str
    score: float | None = None
    timestamp: str | None = None
    source: str = ""  # "session", "knowledge", "core", "preferences"


@dataclass
class MemorySearch:
    """Hybrid search engine for MemoryBrick.

    Indexes session logs and knowledge facts into SQLite FTS5 for BM25
    keyword search. Optional vector embeddings enable semantic search
    with 7:3 weighted fusion.
    """

    store: MemoryStore
    _indexed: bool = False

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search memory with BM25 + optional vector fusion.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results.

        Returns:
            List of MemoryEntry sorted by relevance score (descending).
        """
        self._ensure_index()
        results = self._bm25_search(query)

        # Apply time decay (except for evergreen entries)
        results = self._apply_time_decay(results)

        # Apply MMR reranking for diversity
        results = self._mmr_rerank(results, top_k)

        return results[:top_k]

    def _ensure_index(self) -> None:
        """Create FTS5 index if not exists and populate with fresh data."""
        db = self.store.get_db()
        db.execute(
            "CREATE TABLE IF NOT EXISTS memory_index ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  content TEXT NOT NULL,"
            "  source TEXT NOT NULL,"
            "  timestamp TEXT NOT NULL,"
            "  is_evergreen INTEGER DEFAULT 0"
            ")"
        )
        db.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts "
            "USING fts5(content, source, timestamp, content=memory_index, content_rowid=id)"
        )

        # Triggers to keep FTS in sync
        db.executescript("""
            CREATE TRIGGER IF NOT EXISTS mem_ai AFTER INSERT ON memory_index BEGIN
                INSERT INTO memory_fts(rowid, content, source, timestamp)
                VALUES (new.id, new.content, new.source, new.timestamp);
            END;
            CREATE TRIGGER IF NOT EXISTS mem_ad AFTER DELETE ON memory_index BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content, source, timestamp)
                VALUES ('delete', old.id, old.content, old.source, old.timestamp);
            END;
            CREATE TRIGGER IF NOT EXISTS mem_au AFTER UPDATE ON memory_index BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, content, source, timestamp)
                VALUES ('delete', old.id, old.content, old.source, old.timestamp);
                INSERT INTO memory_fts(rowid, content, source, timestamp)
                VALUES (new.id, new.content, new.source, new.timestamp);
            END;
        """)

        if not self._indexed:
            self._populate_index(db)
            self._indexed = True

        db.commit()

    def _populate_index(self, db) -> None:
        """Populate FTS5 index from session logs and knowledge facts."""
        # Clear existing data
        db.execute("DELETE FROM memory_index")

        # Index session entries
        for entry in self.store.iter_session_entries():
            content = entry.get("content", "")
            if content:
                db.execute(
                    "INSERT INTO memory_index (content, source, timestamp, is_evergreen) "
                    "VALUES (?, ?, ?, 0)",
                    (content, "session", entry.get("timestamp", "")),
                )

        # Index knowledge facts
        for fact in self.store.iter_knowledge_facts():
            db.execute(
                "INSERT INTO memory_index (content, source, timestamp, is_evergreen) "
                "VALUES (?, ?, ?, 0)",
                (fact.get("fact", ""), "knowledge", fact.get("timestamp", "")),
            )

        # Index core memory (evergreen)
        core = self.store.get_core_memory()
        if core:
            db.execute(
                "INSERT INTO memory_index (content, source, timestamp, is_evergreen) "
                "VALUES (?, ?, ?, 1)",
                (core, "core", datetime.now(timezone.utc).isoformat()),
            )

        # Index preferences (evergreen)
        prefs = self.store.get_preferences()
        if prefs:
            db.execute(
                "INSERT INTO memory_index (content, source, timestamp, is_evergreen) "
                "VALUES (?, ?, ?, 1)",
                (prefs, "preferences", datetime.now(timezone.utc).isoformat()),
            )

    def _bm25_search(self, query: str) -> list[MemoryEntry]:
        """BM25 keyword search via SQLite FTS5."""
        db = self.store.get_db()
        self._ensure_index()

        # Escape special FTS5 characters
        safe_query = self._escape_fts5_query(query)

        try:
            cursor = db.execute(
                "SELECT m.content, m.source, m.timestamp, m.is_evergreen, "
                "bm25(memory_fts, 1.0, 0.75) AS rank "
                "FROM memory_fts f "
                "JOIN memory_index m ON f.rowid = m.id "
                "WHERE memory_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT 20",
                (safe_query,),
            )
            results: list[MemoryEntry] = []
            for row in cursor.fetchall():
                content, source, timestamp, is_evergreen, rank = row
                # BM25 returns negative scores (lower = better), invert to positive
                score = 1.0 / (1.0 + abs(rank)) if rank is not None else 0.5
                results.append(MemoryEntry(
                    content=content,
                    score=score,
                    timestamp=timestamp,
                    source=source,
                ))
            return results
        except Exception:
            # FTS5 query failed — fall back to simple LIKE search
            return self._fallback_search(query)

    def _fallback_search(self, query: str) -> list[MemoryEntry]:
        """Simple LIKE-based search fallback when FTS5 fails."""
        db = self.store.get_db()
        like_query = f"%{query}%"
        cursor = db.execute(
            "SELECT content, source, timestamp, is_evergreen "
            "FROM memory_index "
            "WHERE content LIKE ? "
            "LIMIT 20",
            (like_query,),
        )
        results: list[MemoryEntry] = []
        for row in cursor.fetchall():
            content, source, timestamp, is_evergreen = row
            # Simple heuristic: shorter content = more relevant match
            score = 1.0 / (1.0 + len(content) / 100.0)
            results.append(MemoryEntry(
                content=content,
                score=score,
                timestamp=timestamp,
                source=source,
            ))
        return results

    @staticmethod
    def _escape_fts5_query(query: str) -> str:
        """Escape special FTS5 characters and format for prefix matching."""
        # FTS5 special chars: * " - ( ) :
        escaped = re.sub(r'([*"\-():])', r'\\\1', query)
        # Add prefix wildcard to last word for partial matching
        words = escaped.strip().split()
        if words:
            words[-1] = words[-1] + "*"
        return " ".join(words)

    def _apply_time_decay(self, results: list[MemoryEntry]) -> list[MemoryEntry]:
        """Apply time decay: score × e^(-λ × days_since).

        Evergreen entries (preferences, core memory) are exempt.
        """
        now = datetime.now(timezone.utc)
        for entry in results:
            if entry.source in ("core", "preferences"):
                continue  # Evergreen — no decay
            if entry.timestamp and entry.score is not None:
                try:
                    ts = datetime.fromisoformat(entry.timestamp.replace("Z", "+00:00"))
                    days_since = (now - ts).total_seconds() / 86400.0
                    decay = math.exp(-DECAY_LAMBDA * days_since)
                    entry.score *= decay
                except (ValueError, TypeError):
                    pass
        return results

    def _mmr_rerank(self, results: list[MemoryEntry], top_k: int) -> list[MemoryEntry]:
        """Maximal Marginal Relevance reranking for diversity.

        Balances relevance with novelty to avoid returning near-duplicate
        results. Lambda=0.7 weights relevance higher than diversity.
        """
        if len(results) <= top_k:
            return sorted(results, key=lambda r: r.score or 0, reverse=True)

        LAMBDA = 0.7  # relevance vs diversity weight
        selected: list[MemoryEntry] = []
        remaining = list(results)

        # First pick: highest score
        remaining.sort(key=lambda r: r.score or 0, reverse=True)
        selected.append(remaining.pop(0))

        # Greedy MMR selection
        while len(selected) < top_k and remaining:
            best_idx = 0
            best_score = -float("inf")
            for i, candidate in enumerate(remaining):
                relevance = candidate.score or 0
                # Diversity penalty: max similarity to any selected
                max_sim = max(
                    self._jaccard_similarity(candidate.content, s.content)
                    for s in selected
                )
                mmr = LAMBDA * relevance - (1 - LAMBDA) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            selected.append(remaining.pop(best_idx))

        return selected

    @staticmethod
    def _jaccard_similarity(a: str, b: str) -> float:
        """Simple Jaccard similarity on word sets for MMR diversity."""
        set_a = set(a.lower().split())
        set_b = set(b.lower().split())
        if not set_a or not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union)

    def rebuild_index(self) -> None:
        """Force rebuild of the FTS5 search index."""
        db = self.store.get_db()
        db.execute("DELETE FROM memory_index")
        self._indexed = False
        self._ensure_index()
        db.commit()