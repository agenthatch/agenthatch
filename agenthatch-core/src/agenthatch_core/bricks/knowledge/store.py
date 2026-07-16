"""KnowledgeStore — generic vector + keyword store for RAG (v1.0.0).

Decoupled from MemorySearch to support the KnowledgeBaseBrick runtime.
Unlike MemorySearch (which indexes the agent's own session logs and
memories), KnowledgeStore holds external authoritative knowledge provided
by the user at hatch time.

Three-layer retrieval (inspired by OpenClaw):
  1. BM25 via SQLite FTS5 (always available, zero dependencies)
  2. BM25 + embedding fusion (7:3 weighted, when model configured)
  3. LLM re-rank (optional, for precision-critical queries)

The store is a single SQLite database file — persist/load is just
opening/closing the file.  No separate serialization needed.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger("agenthatch")

# Embedding model download timeout (seconds).
# If exceeded, embedding search is disabled gracefully → keyword-only mode.
_EMBEDDER_TIMEOUT = 60


@dataclass
class KBDocument:
    """A single document chunk in the knowledge base."""
    doc_id: str               # unique identifier (e.g. "geography.md#3")
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # metadata keys: source, chunk_index, content_type, tags, importance


@dataclass
class KBSearchResult:
    """A single search result from the knowledge base."""
    doc_id: str
    content: str
    score: float              # 0.0–1.0, higher = more relevant
    metadata: dict[str, Any] = field(default_factory=dict)
    match_source: str = "keyword"  # "keyword" | "embedding" | "hybrid" | "reranked"


class KnowledgeStore:
    """Generic vector + keyword store for RAG retrieval.

    Usage (build-time, during hatch Phase 3.5)::

        store = KnowledgeStore(Path("agent/knowledge"))
        store.add_document("doc1#0", "content...", {"source": "doc1.md"})
        store.build_index()
        store.close()

    Usage (run-time, in KnowledgeBaseBrick)::

        store = KnowledgeStore(Path("agent/knowledge"))
        store.load()
        results = store.search("query", top_k=5)
        store.close()

    The SQLite database file is the persistent index.  ``build_index()``
    finalizes the FTS5 table; ``load()`` opens an existing database.
    """

    def __init__(
        self,
        store_path: Path | str,
        embedding_model: str = "all-MiniLM-L6-v2",
        enable_llm_rerank: bool = True,
    ):
        """Initialize the knowledge store.

        Args:
            store_path: Directory path for the SQLite database.
                        The database file is ``{store_path}/kb_index.db``.
            embedding_model: sentence-transformers model name for semantic
                             search.  Download is lazy and may fail gracefully.
            enable_llm_rerank: If True, ``search()`` will call the injected
                                ``rerank_fn`` for final re-ranking.
        """
        self._path = Path(store_path)
        self._path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._path / "kb_index.db"

        self._embedding_model_name = embedding_model
        self._enable_llm_rerank = enable_llm_rerank
        self._rerank_fn: Callable[[str, list[KBSearchResult], int], list[KBSearchResult]] | None = None

        # Lazy-init caches
        self._db: sqlite3.Connection | None = None
        self._embedder: Any = None
        self._embedder_disabled: bool = False
        self._index_built: bool = False

        # v0.9.17: thread-local connections for parallel retrieve.
        # ConversationLoop dispatches tool calls concurrently; SQLite
        # connections can't cross threads by default.  Each worker
        # thread gets its own read-only connection, while the main
        # thread keeps its (write-capable) connection for build_index.
        self._thread_local = threading.local()
        self._write_lock = threading.Lock()
        # v1.0.1 (R2-H3): Track all live connections keyed by thread
        # ident so ``close()`` can reap worker-thread connections too
        # (fixes resource leak + stale thread-local cache returning
        # closed connections).  We can't use ``weakref.WeakSet``
        # because ``sqlite3.Connection`` doesn't support weakref —
        # instead we periodically prune entries whose owning thread
        # has exited (see ``_prune_dead_conns()``).  Without pruning,
        # long-running services that spawn-and-die threads would
        # accumulate dead connections forever.
        self._all_conns: dict[int, sqlite3.Connection] = {}
        self._conn_lock = threading.Lock()
        # v1.0.1 (R2-H2): Generation counter — incremented on close().
        # Worker threads cache their connection + generation in
        # thread-local; on next _get_db() they compare cached
        # generation to current and discard stale connections instead
        # of returning a closed one (ProgrammingError).
        self._conn_generation: int = 0
        # v1.0.1 (R2-H3): Prune threshold — when dict size exceeds
        # this, scan ``threading.enumerate()`` and discard entries
        # whose owning thread has exited.
        self._conn_prune_threshold: int = 32

    # ── Database lifecycle ──────────────────────────────────────────

    def _get_db(self) -> sqlite3.Connection:
        """Get or create a thread-local SQLite connection.

        SQLite connections cannot be shared across threads by default.
        The ConversationLoop dispatches tool calls in parallel, so each
        worker thread gets its own connection.  The main thread keeps
        its connection cached as ``self._db`` for write paths
        (``add_documents``, ``build_index``).

        v1.0.1 (R2-H2): Thread-local cache entries carry a generation
        counter.  ``close()`` bumps the counter; on next ``_get_db()``
        call, worker threads see their cached generation is stale and
        discard the closed connection instead of returning it (which
        would raise ``ProgrammingError: Cannot operate on a closed
        database``).
        """
        # Fast path: thread-local cache hit, but verify generation
        # v1.0.1: Annotate explicitly so mypy --strict doesn't flag
        # ``Returning Any from function declared to return Connection``.
        conn: sqlite3.Connection | None = getattr(
            self._thread_local, "conn", None
        )
        cached_gen: int = getattr(
            self._thread_local, "generation", -1
        )
        if conn is not None and cached_gen == self._conn_generation:
            return conn

        # Main thread: prefer the cached self._db (write-capable)
        is_main = threading.current_thread() is threading.main_thread()
        if (
            is_main
            and self._db is not None
            and cached_gen == self._conn_generation
        ):
            return self._db

        # Create a new connection for this thread
        conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        # v1.0.1: Set busy_timeout so concurrent reads during writes
        # wait up to 5s instead of immediately failing with
        # "database is locked".
        conn.execute("PRAGMA busy_timeout = 5000")
        # v1.0.1 (R2b-M2): Enable WAL (Write-Ahead Logging) journal mode.
        # In the default rollback-journal mode, a writer blocks ALL
        # readers (even with busy_timeout, they just wait and retry).
        # WAL allows concurrent reads during writes — critical for RAG
        # workloads where reads vastly outnumber writes (the index is
        # built once at hatch time, then only reads happen at runtime).
        # WAL also survives crashes better (no torn writes).
        # Setting is persistent per-database-file, but we set it on every
        # new connection to be safe (idempotent, ~0 cost).
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            # synchronous=NORMAL is the recommended pairing for WAL —
            # fsyncs only at checkpoint, not every commit.  Slightly
            # less durable than FULL but 2-3x faster writes.
            conn.execute("PRAGMA synchronous = NORMAL")
        except sqlite3.OperationalError as e:
            # WAL may fail on network filesystems (NFS) — fall back to
            # default journal mode and log.
            logger.warning("KnowledgeStore: cannot enable WAL mode (%s) — using default", e)
        # Worker threads are read-only — schema is already initialized
        # by the main thread at build time.
        if is_main:
            self._init_schema_for_conn(conn)
            self._db = conn
        else:
            # v1.0.1 (R2b-M3): Force worker connections to query_only.
            # The comment above says "Worker threads are read-only" but
            # previously this was only enforced by convention — a buggy
            # worker could call add_documents() on its connection and
            # race with the main thread's writes.  PRAGMA query_only
            # enforces this at the SQLite layer (raises OperationalError
            # on any INSERT/UPDATE/DELETE).
            try:
                conn.execute("PRAGMA query_only = ON")
            except sqlite3.OperationalError as e:
                logger.warning("KnowledgeStore: cannot set query_only (%s)", e)
        # v1.0.1 (R2-H3): Track for cleanup in close(); dict keyed
        # by thread ident so we can prune dead-thread entries.
        with self._conn_lock:
            self._all_conns[threading.get_ident()] = conn
            # Periodically prune entries from exited worker threads
            # to prevent unbounded growth in long-running services.
            if len(self._all_conns) > self._conn_prune_threshold:
                self._prune_dead_conns()
        # Cache in thread-local with current generation so we can
        # detect staleness after close().
        self._thread_local.conn = conn
        self._thread_local.generation = self._conn_generation
        return conn

    def _init_schema_for_conn(self, db: sqlite3.Connection) -> None:
        """Initialize schema on a given connection.

        Extracted from ``_init_schema`` so worker threads can re-use
        the exact same DDL when they create their own connection.
        """
        db.executescript("""
            CREATE TABLE IF NOT EXISTS kb_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id TEXT UNIQUE NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT NOT NULL DEFAULT '{}',
                embedding BLOB,
                created_at TEXT NOT NULL
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS kb_fts
            USING fts5(content, content=kb_documents, content_rowid=id);

            CREATE TRIGGER IF NOT EXISTS kb_ai AFTER INSERT ON kb_documents BEGIN
                INSERT INTO kb_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
            CREATE TRIGGER IF NOT EXISTS kb_ad AFTER DELETE ON kb_documents BEGIN
                INSERT INTO kb_fts(kb_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
            END;
            CREATE TRIGGER IF NOT EXISTS kb_au AFTER UPDATE ON kb_documents BEGIN
                INSERT INTO kb_fts(kb_fts, rowid, content)
                VALUES ('delete', old.id, old.content);
                INSERT INTO kb_fts(rowid, content)
                VALUES (new.id, new.content);
            END;
        """)
        db.commit()

    def _init_schema(self) -> None:
        """Create tables if not exist (legacy entry — delegates to _init_schema_for_conn)."""
        assert self._db is not None
        self._init_schema_for_conn(self._db)

    def _prune_dead_conns(self) -> None:
        """Drop connections whose owning thread has exited (v1.0.1 R2-H3).

        ``sqlite3.Connection`` doesn't support ``weakref``, so we can't
        rely on GC to drop entries when their owning thread exits.
        Instead we scan ``threading.enumerate()`` to find live idents
        and discard entries whose ident isn't present.

        Caller must hold ``self._conn_lock``.  Safe to call from any
        thread — only closes connections from threads that no longer
        appear in ``threading.enumerate()``, and dead threads can't be
        using their connection.

        Note on ident reuse: if a dead thread's ident has been reused
        by a new live thread that hasn't yet called ``_get_db()``,
        ``_all_conns[tid]`` still points at the dead thread's
        connection.  We won't close it (because ``tid`` is in
        ``alive_idents``).  When the new thread eventually calls
        ``_get_db()`` it overwrites the entry, releasing the dead
        thread's connection for GC.  Self-healing, no leak.
        """
        alive_idents: set[int] = {
            t.ident for t in threading.enumerate() if t.ident is not None
        }
        dead_idents = [
            tid for tid in list(self._all_conns.keys())
            if tid not in alive_idents
        ]
        for tid in dead_idents:
            conn = self._all_conns.pop(tid, None)
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.ProgrammingError:
                    pass  # Already closed (e.g. by GC of worker thread)

    def load(self) -> None:
        """Open an existing pre-built index (run-time use).

        Verifies the FTS5 table exists and is populated.

        v1.0.1 (R2-C1): Also lazy-init the embedder so runtime
        ``search()`` can perform embedding fusion.  Previously
        ``_ensure_embedder()`` was only called from ``build_index()``
        (build-time path), leaving the runtime ``self._embedder``
        perpetually ``None`` — the ``search()`` embedding branch
        silently no-op'd and all queries degraded to keyword-only.
        """
        db = self._get_db()
        cursor = db.execute("SELECT COUNT(*) FROM kb_documents")
        count = cursor.fetchone()[0]
        if count == 0:
            logger.warning("KnowledgeStore.load(): index is empty (0 documents)")
        else:
            logger.info("KnowledgeStore.load(): %d documents loaded", count)

        # v1.0.1: Verify FTS5 table exists — a corrupted index would
        # otherwise fail late in _bm25_search with OperationalError.
        fts_check = db.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='kb_fts'"
        ).fetchone()
        if fts_check is None:
            raise sqlite3.OperationalError(
                "kb_fts table missing — index corrupted or not built. "
                "Re-run hatch to rebuild."
            )
        self._index_built = True

        # v1.0.1 (R2-C1): Lazy-init embedder at runtime load so
        # ``search()`` can perform hybrid BM25 + embedding fusion.
        # Skipped if the store was constructed with embedding disabled
        # or if the model download fails (graceful keyword-only fallback).
        self._ensure_embedder()

    def close(self) -> None:
        """Close all database connections (main + worker threads).

        v1.0.1: Previously only closed ``self._db`` (main thread),
        leaving worker-thread connections to be GC'd and the
        thread-local cache pointing at a stale closed connection.
        Now tracks all connections in ``_all_conns`` and closes
        them all, then clears the thread-local cache.

        v1.0.1 (R2-H2): Bumps ``_conn_generation`` so worker threads
        still holding a thread-local cache entry see it's stale on
        next ``_get_db()`` and create a fresh connection instead of
        returning the now-closed one (ProgrammingError).

        v1.0.1 (R2-H3): ``_all_conns`` is now a ``dict[int, Connection]``
        keyed by thread ident; snapshotting ``list(self._all_conns.values())``
        produces a strong-ref list so iteration is stable even
        though ``conn.close()`` doesn't mutate the dict.
        """
        with self._conn_lock:
            # Snapshot values so iteration is stable.
            conns_snapshot = list(self._all_conns.values())
            for conn in conns_snapshot:
                try:
                    conn.close()
                except sqlite3.ProgrammingError:
                    pass  # Already closed (e.g. by GC of worker thread)
            self._all_conns.clear()
            self._db = None
            # v1.0.1 (R2-H2): Bump generation so worker threads'
            # thread-local caches become stale and get refreshed
            # on next _get_db() call.
            self._conn_generation += 1
        # v1.0.1: Clear current thread's thread-local cache so the
        # next _get_db() creates a fresh connection instead of
        # returning the now-closed one.
        if hasattr(self._thread_local, "conn"):
            try:
                del self._thread_local.conn
            except AttributeError:
                pass

    # ── Document management ─────────────────────────────────────────

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a document chunk to the store (build-time).

        Args:
            doc_id: Unique identifier (e.g. "geography.md#3").
            content: The text content of this chunk.
            metadata: Optional metadata (source, chunk_index, tags, etc.).
        """
        from datetime import datetime, timezone
        db = self._get_db()
        meta_json = json.dumps(metadata or {}, ensure_ascii=False)
        now = datetime.now(timezone.utc).isoformat()
        db.execute(
            "INSERT OR REPLACE INTO kb_documents (doc_id, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?)",
            (doc_id, content, meta_json, now),
        )
        db.commit()

    def add_documents(self, documents: list[KBDocument]) -> None:
        """Batch-add multiple documents (build-time)."""
        from datetime import datetime, timezone
        db = self._get_db()
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (d.doc_id, d.content, json.dumps(d.metadata, ensure_ascii=False), now)
            for d in documents
        ]
        db.executemany(
            "INSERT OR REPLACE INTO kb_documents (doc_id, content, metadata, created_at) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        db.commit()

    def build_index(self) -> None:
        """Finalize the FTS5 index after all documents are added (build-time).

        Also pre-computes embeddings if the embedder is available.
        """
        db = self._get_db()
        # FTS5 is kept in sync via triggers, so just verify row count
        cursor = db.execute("SELECT COUNT(*) FROM kb_documents")
        count = cursor.fetchone()[0]
        logger.info("KnowledgeStore.build_index(): %d documents indexed", count)

        # Optionally compute embeddings for all documents
        self._ensure_embedder()
        if self._embedder is not None and not self._embedder_disabled:
            self._compute_all_embeddings()

        db.commit()
        self._index_built = True

    # ── Embedding support (optional) ────────────────────────────────

    def _ensure_embedder(self) -> None:
        """Lazy-init sentence-transformer for embedding search.

        If the model cannot be downloaded (offline/network issues),
        embedding search is disabled gracefully — keyword-only mode.
        """
        if self._embedder is not None or self._embedder_disabled:
            return

        try:
            import io as _io
            import sys as _sys
            from sentence_transformers import SentenceTransformer

            embedder_result: list[Any] = [None]
            embedder_error: list[Any] = [None]

            def _load() -> None:
                try:
                    hf_log = logging.getLogger("huggingface_hub")
                    hf_level = hf_log.level
                    hf_log.setLevel(logging.ERROR)
                    _stderr = _sys.stderr
                    _sys.stderr = _io.StringIO()
                    try:
                        embedder_result[0] = SentenceTransformer(self._embedding_model_name)
                    finally:
                        _sys.stderr = _stderr
                        hf_log.setLevel(hf_level)
                except Exception as e:
                    embedder_error[0] = e

            t = threading.Thread(target=_load, daemon=True)
            t.start()
            t.join(timeout=_EMBEDDER_TIMEOUT)
            if t.is_alive():
                logger.warning(
                    "SentenceTransformer download timed out (%ds). "
                    "Embedding search disabled.", _EMBEDDER_TIMEOUT
                )
                self._embedder_disabled = True
                return
            if embedder_error[0]:
                # v1.0.1 (R4-V12): Distinguish SSL/certificate errors
                # (common on macOS Python 3.11+, fixable by the user)
                # from generic network failures, so the user knows what
                # to do.  The generic "Cannot send a request, as the
                # client has been closed" message from httpx is opaque.
                err_msg = str(embedder_error[0])
                err_type = type(embedder_error[0]).__name__
                if (
                    "CERTIFICATE" in err_msg.upper()
                    or "SSL" in err_msg.upper()
                    or "client has been closed" in err_msg
                ):
                    logger.warning(
                        "SentenceTransformer model download failed (%s: %s). "
                        "This is typically an SSL certificate issue on macOS — "
                        "run '/Applications/Python 3.x/Install Certificates.command' "
                        "or set SSL_CERT_FILE to certifi's bundle. "
                        "Embedding search disabled; BM25 keyword search still active.",
                        err_type, err_msg,
                    )
                else:
                    logger.warning(
                        "SentenceTransformer download failed (%s: %s). "
                        "Embedding search disabled (keyword-only mode).",
                        err_type, err_msg,
                    )
                self._embedder_disabled = True
                return
            self._embedder = embedder_result[0]
            logger.info("KnowledgeStore: embedder loaded (%s)", self._embedding_model_name)
        except ImportError:
            logger.info(
                "sentence-transformers not installed. "
                "Embedding search disabled (keyword-only mode)."
            )
            self._embedder_disabled = True

    def _compute_all_embeddings(self) -> None:
        """Compute and store embeddings for all documents."""
        if self._embedder is None:
            return
        db = self._get_db()
        cursor = db.execute("SELECT doc_id, content FROM kb_documents WHERE embedding IS NULL")
        rows = cursor.fetchall()
        if not rows:
            return
        import numpy as np

        # v1.0.1: Batch encode for performance — 10-20x faster on
        # large KBs (1000+ chunks).  Previous per-row encode() was
        # O(n) LLM forward passes; batch_size=32 lets the embedder
        # parallelize GPU/CPU work.
        contents = [row["content"] for row in rows]
        embeddings = self._embedder.encode(
            contents, batch_size=32, normalize_embeddings=True
        )
        for row, emb in zip(rows, embeddings):
            emb_bytes = np.array(emb, dtype=np.float32).tobytes()
            db.execute(
                "UPDATE kb_documents SET embedding = ? WHERE doc_id = ?",
                (emb_bytes, row["doc_id"]),
            )
        db.commit()
        logger.info("KnowledgeStore: computed embeddings for %d documents", len(rows))

    # ── Search ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        alpha: float = 0.7,
        enable_rerank: bool | None = None,
    ) -> list[KBSearchResult]:
        """Hybrid search: BM25 + embedding + optional LLM re-rank.

        Args:
            query: Natural language search query.
            top_k: Maximum number of results to return.
            alpha: Keyword weight (0.0=embedding-only, 1.0=keyword-only).
                   Default 0.7 favors keyword (BM25) for precise matching.
            enable_rerank: Override the store's default rerank setting.
                          Set False to skip LLM re-rank for speed.

        Returns:
            Ranked list of KBSearchResult (highest score first).
        """
        # v1.0.1 (R3-M5): Clamp negative/zero ``top_k`` to 1.  A
        # negative ``top_k`` would previously produce ``top_k * 2 = -2``
        # passed as ``LIMIT ?`` — SQLite treats a negative LIMIT as
        # "no limit", returning every matching row while the final
        # ``fused[:top_k]`` slice silently dropped everything (empty
        # list).  Clamping at the entry point keeps the contract
        # "always returns at most top_k results" intact and surfaces
        # the bad input via a warning.
        if top_k <= 0:
            logger.warning(
                "KnowledgeStore.search: top_k=%d clamped to 1 (negative "
                "or zero top_k would return empty results).",
                top_k,
            )
            top_k = 1

        if not self._index_built:
            self.load()

        # Layer 1: BM25 keyword search
        bm25_results = self._bm25_search(query, top_k * 2)

        # Layer 2: Embedding semantic search (if available)
        emb_results: list[KBSearchResult] = []
        if not self._embedder_disabled and self._embedder is not None:
            emb_results = self._embedding_search(query, top_k * 2)

        # Fuse results
        fused = self._fuse_results(bm25_results, emb_results, alpha)

        # Layer 3: LLM Re-rank (optional)
        do_rerank = self._enable_llm_rerank if enable_rerank is None else enable_rerank
        if do_rerank and self._rerank_fn is not None and len(fused) > top_k:
            fused = self._rerank_fn(query, fused, top_k)
            for r in fused:
                r.match_source = "reranked"

        return fused[:top_k]

    def _bm25_search(self, query: str, limit: int) -> list[KBSearchResult]:
        """BM25 keyword search via SQLite FTS5."""
        db = self._get_db()
        safe_query = self._escape_fts5_query(query)
        if not safe_query.strip():
            return []
        try:
            cursor = db.execute(
                "SELECT d.doc_id, d.content, d.metadata, "
                "bm25(kb_fts) AS rank "
                "FROM kb_fts f "
                "JOIN kb_documents d ON f.rowid = d.id "
                "WHERE kb_fts MATCH ? "
                "ORDER BY rank "
                "LIMIT ?",
                (safe_query, limit),
            )
            results: list[KBSearchResult] = []
            for row in cursor.fetchall():
                # FTS5 bm25() returns *negative* scores — more negative
                # means more relevant (lower rank = better).  We convert
                # to a positive 0–1 score where higher = more relevant
                # so that ``_fuse_results`` (which sorts descending) and
                # embedding scores (cosine similarity, also 0–1 higher=better)
                # share the same polarity.
                #
                # ``abs(rank) / (1 + abs(rank))`` maps ``(-inf, 0]`` to
                # ``[0, 1)`` monotonically increasing in relevance.
                # The previous ``1 / (1 + abs(rank))`` was inverted —
                # the most relevant doc (rank=-9.76) ended up with the
                # lowest score (0.09) and got sorted last.
                rank = row["rank"]
                if rank is None:
                    score = 0.5
                else:
                    magnitude = abs(rank)
                    score = magnitude / (1.0 + magnitude)
                metadata = json.loads(row["metadata"]) if row["metadata"] else {}
                results.append(KBSearchResult(
                    doc_id=row["doc_id"],
                    content=row["content"],
                    score=score,
                    metadata=metadata,
                    match_source="keyword",
                ))
            return results
        except sqlite3.OperationalError:
            # FTS5 query failed — fall back to LIKE search
            return self._fallback_search(query, limit)

    def _fallback_search(self, query: str, limit: int) -> list[KBSearchResult]:
        """Simple LIKE-based search fallback when FTS5 fails."""
        db = self._get_db()
        # v1.0.1: Escape LIKE wildcards (%, _) and backslash so user
        # queries like "100%" or "file_name" don't act as wildcards.
        escaped = (
            query.replace("\\", "\\\\")
                 .replace("%", "\\%")
                 .replace("_", "\\_")
        )
        like_query = f"%{escaped}%"
        cursor = db.execute(
            "SELECT doc_id, content, metadata FROM kb_documents "
            "WHERE content LIKE ? ESCAPE '\\' LIMIT ?",
            (like_query, limit),
        )
        results: list[KBSearchResult] = []
        for row in cursor.fetchall():
            score = 1.0 / (1.0 + len(row["content"]) / 100.0)
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            results.append(KBSearchResult(
                doc_id=row["doc_id"],
                content=row["content"],
                score=score,
                metadata=metadata,
                match_source="keyword",
            ))
        return results

    def _embedding_search(self, query: str, limit: int) -> list[KBSearchResult]:
        """Embedding cosine similarity search."""
        import numpy as np
        db = self._get_db()
        query_emb = self._embedder.encode(query, normalize_embeddings=True)

        cursor = db.execute(
            "SELECT doc_id, content, metadata, embedding FROM kb_documents "
            "WHERE embedding IS NOT NULL"
        )
        scored: list[tuple[float, KBSearchResult]] = []
        for row in cursor.fetchall():
            emb_bytes = row["embedding"]
            if emb_bytes is None:
                continue
            doc_emb = np.frombuffer(emb_bytes, dtype=np.float32)
            cos_sim = float(np.dot(query_emb, doc_emb))
            metadata = json.loads(row["metadata"]) if row["metadata"] else {}
            result = KBSearchResult(
                doc_id=row["doc_id"],
                content=row["content"],
                score=cos_sim,
                metadata=metadata,
                match_source="embedding",
            )
            scored.append((cos_sim, result))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:limit]]

    @staticmethod
    def _fuse_results(
        bm25_results: list[KBSearchResult],
        emb_results: list[KBSearchResult],
        alpha: float,
    ) -> list[KBSearchResult]:
        """Fuse BM25 and embedding results with weighted scoring.

        final_score = alpha * keyword_score + (1-alpha) * embedding_score

        v1.0.1: No longer mutates input ``bm25_results`` scores —
        creates new KBSearchResult objects so callers can reuse the
        original lists (e.g. for logging or debugging) without
        seeing normalized scores.
        """
        if not emb_results:
            # Keyword-only mode
            return sorted(bm25_results, key=lambda r: r.score, reverse=True)
        if not bm25_results:
            return sorted(emb_results, key=lambda r: r.score, reverse=True)

        # Normalize BM25 scores to 0-1 (without mutating inputs)
        max_bm25 = max((r.score for r in bm25_results), default=1.0) or 1.0
        normalized_bm25: dict[str, float] = {
            r.doc_id: r.score / max_bm25 for r in bm25_results
        }

        # Fuse by doc_id
        fused: dict[str, KBSearchResult] = {}
        for r in bm25_results:
            fused[r.doc_id] = KBSearchResult(
                doc_id=r.doc_id,
                content=r.content,
                score=alpha * normalized_bm25[r.doc_id],
                metadata=r.metadata,
                match_source="keyword",
            )
        for r in emb_results:
            if r.doc_id in fused:
                fused[r.doc_id].score += (1.0 - alpha) * r.score
                fused[r.doc_id].match_source = "hybrid"
            else:
                fused[r.doc_id] = KBSearchResult(
                    doc_id=r.doc_id,
                    content=r.content,
                    score=(1.0 - alpha) * r.score,
                    metadata=r.metadata,
                    match_source="embedding",
                )
        return sorted(fused.values(), key=lambda r: r.score, reverse=True)

    @staticmethod
    def _escape_fts5_query(query: str) -> str:
        """Escape special FTS5 characters and format for prefix matching.

        Uses OR semantics for better recall (RAG should find partial matches).
        BM25 still ranks documents with more matching terms higher.

        Hyphens are replaced with spaces *before* tokenization — FTS5 treats
        ``-`` as the NOT operator in query syntax, so ``wind-rider*`` would
        be parsed as ``wind NOT rider*`` and match nothing in documents that
        contain both ``wind`` and ``rider``.  Since the unicode61 tokenizer
        already splits hyphenated words at index time, splitting them at
        query time keeps the query consistent with the index.

        v1.0.1: Also escape backslash — FTS5 treats ``\\`` as an escape
        prefix, so an unescaped backslash (e.g. Windows path
        ``C:\\Users``) would silently truncate the query at the ``\\``.
        """
        # Replace hyphens with spaces first — they are FTS5 NOT operators
        # in query syntax AND separators in the unicode61 tokenizer.
        # Also escape remaining special chars: * " ( ) : ^ \
        cleaned = query.replace("-", " ")
        escaped = re.sub(r'([*"():^\\])', r'\\\1', cleaned)
        words = escaped.strip().split()
        if not words:
            return ""
        # Add prefix wildcard to each word for partial matching
        words = [w + "*" for w in words if w]
        # Use OR for better recall — partial term matches still get scored
        return " OR ".join(words)

    # ── LLM Re-rank injection ───────────────────────────────────────

    def set_rerank_fn(
        self,
        fn: Callable[[str, list[KBSearchResult], int], list[KBSearchResult]],
    ) -> None:
        """Inject an LLM re-rank callable.

        The callable receives (query, candidates, top_k) and returns a
        re-ordered list of at most ``top_k`` results.
        """
        self._rerank_fn = fn

    # ── Stats ───────────────────────────────────────────────────────

    def get_stats(self) -> dict[str, Any]:
        """Return store statistics."""
        db = self._get_db()
        cursor = db.execute("SELECT COUNT(*) FROM kb_documents")
        doc_count = cursor.fetchone()[0]
        size_bytes = self._db_path.stat().st_size if self._db_path.exists() else 0
        return {
            "total_documents": doc_count,
            "index_size_bytes": size_bytes,
            "embedding_enabled": self._embedder is not None and not self._embedder_disabled,
            "llm_rerank_enabled": self._enable_llm_rerank and self._rerank_fn is not None,
        }

    def __enter__(self) -> KnowledgeStore:
        self._get_db()
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def __del__(self) -> None:
        """Best-effort cleanup of SQLite connections on GC (v1.0.1 R3-M7).

        ``sqlite3.Connection`` doesn't release file handles promptly when
        garbage-collected, so a KnowledgeStore that goes out of scope
        without an explicit ``close()`` can leave WAL/SHM sidecar files
        on disk and an open file descriptor.  We swallow all errors here
        because ``__del__`` runs during interpreter shutdown where
        modules and threads may already be torn down (any attribute
        access can raise ``AttributeError``; ``close()`` may raise
        ``sqlite3.ProgrammingError`` if a thread already closed the
        connection).

        Note: this is a safety net, not a primary cleanup path.  Callers
        should still prefer ``with KnowledgeStore(...) as store:`` or
        explicit ``store.close()`` for deterministic lifecycle.
        """
        try:
            self.close()
        except Exception:
            pass  # GC/interpreter-shutdown tolerant


