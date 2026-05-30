"""skillhouse.json — Minimal index repository.

Hybrid search: BM25 keyword (triggers) + embedding (satisfies).
Draws from semantic-router's HybridRouter double-layer matching.
alpha=0.7: keyword match weighted higher for precise trigger matching.

Operations: load, save, search, add_entry, remove_entry, list_all,
            find_provider (O(1) dict lookup), topological_sort (O(V+E) Kahn).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger("agenthatch")


@dataclass
class SearchResult:
    """A single search result from skillhouse.json."""
    skill_id: str
    display_name: str
    summary: str
    score: float           # 0.0–1.0
    match_source: str      # "keyword" | "embedding" | "hybrid"


class SkillhouseIndex:
    """skillhouse.json reader/writer + hybrid search engine.

    Search strategy (alpha=0.7 default):
        final_score = alpha * keyword_score + (1-alpha) * embedding_score

    Usage:
        idx = SkillhouseIndex()
        results = idx.search("weather forecast")
        idx.add_entry(skill_id, ahs_spec, ahs_path)
        idx.list_all()
    """

    def __init__(
        self,
        store_path: str | Path = ".agenthatch/skillhouse.json",
        embedding_model_name: str = "all-MiniLM-L6-v2",
    ):
        """Initialize the index.

        Args:
            store_path: Path to skillhouse.json file.
            embedding_model_name: sentence-transformers model for embeddings.
        """
        self._path = Path(store_path).expanduser().resolve()
        self._data: dict[str, Any] = self._load()
        self._embedding_model_name = embedding_model_name

        # Lazy-init caches
        self._bm25: Any = None
        self._bm25_docs: list[str] = []
        self._bm25_ids: list[str] = []
        self._embedder: Any = None
        self._embedder_disabled: bool = False  # Set when model download fails

    # ── I/O ───────────────────────────────────────────────────────────

    def _load(self) -> Any:
        """Load skillhouse.json or return empty skeleton."""
        if self._path.exists():
            try:
                return json.loads(self._path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                logger.warning(f"Failed to load {self._path}: {e}, using empty index")
        return {
            "version": "1.0",
            "entries": {},
            "topology": {"providers": {}, "dependency_graph": {}},
        }

    def _save(self) -> None:
        """Persist index to disk, creating parent dirs if needed."""
        self._data.setdefault("updated_at", datetime.now(UTC).isoformat())
        if "created_at" not in self._data:
            self._data["created_at"] = self._data["updated_at"]

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(
            json.dumps(self._data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    # ── Search Engine Initialization ──────────────────────────────────

    def _ensure_bm25(self) -> None:
        """Lazy-init BM25 index from triggers."""
        if self._bm25 is not None:
            return
        from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

        self._bm25_docs = []
        self._bm25_ids = []
        for sid, entry in self._data["entries"].items():
            triggers = entry.get("intent", {}).get("triggers", [])
            if triggers:
                self._bm25_docs.append(" ".join(triggers))
                self._bm25_ids.append(sid)

        if self._bm25_docs:
            self._bm25 = BM25Okapi([doc.split() for doc in self._bm25_docs])
        else:
            self._bm25 = None

    def _ensure_embedder(self) -> None:
        """Lazy-init sentence-transformer for embedding search.

        If the model cannot be downloaded (offline/network issues),
        embedding search is disabled gracefully — keyword-only mode.
        """
        if self._embedder is not None or self._embedder_disabled:
            return
        from sentence_transformers import SentenceTransformer

        try:
            self._embedder = SentenceTransformer(self._embedding_model_name)
        except Exception as e:
            logger.warning(
                f"Failed to load embedding model '{self._embedding_model_name}': {e}. "
                "Embedding search disabled, falling back to keyword-only."
            )
            self._embedder_disabled = True

    # ── Search ────────────────────────────────────────────────────────

    def search(
        self, query: str, top_k: int = 5, alpha: float = 0.7
    ) -> list[SearchResult]:
        """Hybrid search: BM25 keyword + embedding semantic.

        Args:
            query: Natural language search query.
            top_k: Max number of results to return.
            alpha: Keyword weight (0.0=embedding-only, 1.0=keyword-only).

        Returns:
            Ranked list of SearchResult (highest score first).
        """
        self._ensure_bm25()
        self._ensure_embedder()

        scores: dict[str, float] = {}
        match_sources: dict[str, str] = {}

        # Layer 1: BM25 keyword search (triggers)
        if self._bm25 is not None:
            tokenized_query = query.lower().split()
            bm25_scores = self._bm25.get_scores(tokenized_query)
            max_bm25 = float(max(bm25_scores)) if max(bm25_scores) > 0 else 1.0
            for i, sid in enumerate(self._bm25_ids):
                norm_score = float(bm25_scores[i]) / max_bm25
                scores[sid] = scores.get(sid, 0.0) + alpha * norm_score
                match_sources[sid] = "keyword" if norm_score > 0.3 else "hybrid"

        # Layer 2: Embedding semantic search (satisfies)
        if not self._embedder_disabled and self._embedder is not None:
            import numpy as np

            query_emb: np.ndarray = self._embedder.encode(
                query, normalize_embeddings=True
            )
            for sid, entry in self._data["entries"].items():
                stored_emb = entry.get("embedding")
                if not stored_emb:
                    continue
                cos_sim = float(np.dot(query_emb, stored_emb))
                scores[sid] = scores.get(sid, 0.0) + (1.0 - alpha) * cos_sim
                if sid not in match_sources:
                    match_sources[sid] = "embedding"

        # Rank and return
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        results: list[SearchResult] = []
        for sid, score in ranked:
            entry = self._data["entries"][sid]
            results.append(SearchResult(
                skill_id=sid,
                display_name=entry.get("identity", {}).get("display_name", sid),
                summary=entry.get("intent", {}).get("summary", ""),
                score=round(score, 4),
                match_source=match_sources.get(sid, "embedding"),
            ))
        return results

    # ── Index Mutation ────────────────────────────────────────────────

    def add_entry(
        self,
        skill_id: str,
        ahs_spec: Any,  # AHSSpec instance
        ahs_path: str,
    ) -> None:
        """Register a new skill in the index.

        Args:
            skill_id: Unique skill identifier (kebab-case).
            ahs_spec: AHSSpec Pydantic model instance.
            ahs_path: Path to the agenthatch.yaml file.
        """
        entry = {
            "ahs_path": ahs_path,
            "identity": {
                "id": ahs_spec.identity.id,
                "display_name": ahs_spec.identity.display_name,
                "version": ahs_spec.identity.version,
            },
            "intent": {
                "triggers": ahs_spec.intent.triggers,
                "satisfies": ahs_spec.intent.satisfies,
                "summary": ahs_spec.intent.summary,
            },
            "interface": {
                "provides": [
                    {"capability": c.capability, "type": c.type}
                    for c in ahs_spec.interface.provides
                ],
                "requires": [
                    {"capability": c.capability, "type": c.type}
                    for c in ahs_spec.interface.requires
                ],
                "compatible_with": ahs_spec.interface.compatible_with,
            },
            "hash": _compute_ahs_hash(ahs_spec),
        }
        # Generate embedding from satisfies + summary
        entry["embedding"] = self._compute_entry_embedding(entry)  # type: ignore[assignment]

        self._data["entries"][skill_id] = entry
        self._remove_from_topology(skill_id)
        self._update_topology(skill_id, ahs_spec)
        self._bm25 = None  # invalidate BM25 cache
        self._save()

    def _compute_entry_embedding(self, entry: dict[str, Any]) -> list[float]:
        """Compute embedding for a new entry from its intent fields.

        If embedder is unavailable (offline), stores a zero-vector placeholder.
        """
        self._ensure_embedder()
        if self._embedder_disabled or self._embedder is None:
            return []  # Placeholder — keyword-only mode
        text_parts = entry.get("intent", {}).get("satisfies", []) + [
            entry.get("intent", {}).get("summary", "")
        ]
        text = " ".join(text_parts)
        import numpy as np
        emb: np.ndarray = self._embedder.encode(text, normalize_embeddings=True)
        return emb.tolist()  # type: ignore[no-any-return]

    def remove_entry(self, skill_id: str) -> None:
        """Remove a skill from the index."""
        self._data["entries"].pop(skill_id, None)
        self._remove_from_topology(skill_id)
        self._bm25 = None
        self._save()

    # ── Query ─────────────────────────────────────────────────────────

    def list_all(self) -> list[dict[str, Any]]:
        """Return all indexed skills (for `skills` CLI command)."""
        return [
            {
                "id": sid,
                "display_name": e.get("identity", {}).get("display_name", sid),
                "version": e.get("identity", {}).get("version", "?"),
                "summary": e.get("intent", {}).get("summary", ""),
                "ahs_path": e.get("ahs_path", ""),
            }
            for sid, e in self._data["entries"].items()
        ]

    def find_by_name(self, name: str) -> dict[str, Any] | None:
        """Exact match lookup by skill ID. Returns entry dict or None."""
        return self._data.get("entries", {}).get(name)  # type: ignore[no-any-return]

    def get_entry_path(self, skill_id: str) -> str | None:
        """Get the ahs_path for a skill by ID. Returns None if not found."""
        entry = self._data.get("entries", {}).get(skill_id)
        return entry.get("ahs_path") if entry else None

    def register_placeholder(self, skill_id: str, skill_dir: str) -> None:
        """Register a discovered skill with minimal metadata.

        Full metadata is populated later by hatch itself.
        This avoids duplicate scanning on subsequent lookups.

        Pattern: codex loader.rs — skills are registered by path first,
        full metadata loaded on-demand.
        """
        if skill_id in self._data["entries"]:
            return  # already registered — don't overwrite

        self._data["entries"][skill_id] = {
            "ahs_path": str(Path(skill_dir) / "agenthatch.yaml"),
            "identity": {"id": skill_id, "display_name": skill_id},
            "status": "discovered",
        }
        self._save()

    def find_provider(self, capability: str) -> str | None:
        """Topology query: which skill provides this capability? O(1) dict lookup."""
        providers = (
            self._data.get("topology", {}).get("providers", {}).get(capability, [])
        )
        return providers[0] if providers else None

    def topological_sort(self, required_capabilities: list[str]) -> list[str]:
        """O(V+E) Kahn's algorithm topological sort.

        Given a list of required capabilities, return the skill assembly order.
        This is the core dependency resolver for v0.4 brick assembly.

        Args:
            required_capabilities: List of capability names needed.

        Returns:
            Ordered list of skill IDs in dependency order.

        Raises:
            SkillhouseError: If circular dependency is detected.
        """
        from collections import deque

        from agenthatch.exceptions import SkillhouseError

        if not required_capabilities:
            return []

        providers = self._data.get("topology", {}).get("providers", {})
        dep_graph = self._data.get("topology", {}).get("dependency_graph", {})

        # Resolve required capabilities → skill IDs
        needed: set[str] = set()
        for cap in required_capabilities:
            for sid in providers.get(cap, []):
                needed.add(sid)

        if not needed:
            return []

        # Build skill-to-skill adjacency by resolving capability names → skill IDs
        adjacency: dict[str, set[str]] = {sid: set() for sid in needed}
        rev_adjacency: dict[str, set[str]] = {sid: set() for sid in needed}
        for sid in needed:
            for cap in dep_graph.get(sid, []):
                for provider_sid in providers.get(cap, []):
                    if provider_sid in needed:
                        adjacency[sid].add(provider_sid)
                        rev_adjacency.setdefault(provider_sid, set()).add(sid)

        # Build in-degree map
        in_degree: dict[str, int] = {
            sid: len(adjacency[sid]) for sid in needed
        }

        # Kahn's algorithm
        queue = deque(sid for sid, deg in in_degree.items() if deg == 0)
        result: list[str] = []

        while queue:
            node = queue.popleft()
            result.append(node)
            for dependent in rev_adjacency.get(node, ()):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(needed):
            raise SkillhouseError("Circular dependency detected in topology graph")
        return result

    # ── Topology Maintenance ──────────────────────────────────────────

    def _update_topology(self, skill_id: str, ahs_spec: Any) -> None:
        """Update topology graph with new entry's provides/requires."""
        topo = self._data.setdefault(
            "topology", {"providers": {}, "dependency_graph": {}}
        )
        for cap in ahs_spec.interface.provides:
            topo["providers"].setdefault(cap.capability, []).append(skill_id)
        deps = [
            r.capability
            for r in ahs_spec.interface.requires
            if r.capability in topo["providers"]
        ]
        topo["dependency_graph"][skill_id] = deps

    def _remove_from_topology(self, skill_id: str) -> None:
        """Clean up topology entries for removed skill."""
        topo = self._data.get("topology", {})
        for providers_list in topo.get("providers", {}).values():
            if skill_id in providers_list:
                providers_list.remove(skill_id)
        topo.get("dependency_graph", {}).pop(skill_id, None)

    # ── Properties ───────────────────────────────────────────────────

    @property
    def path(self) -> Path:
        """Path to the skillhouse.json file."""
        return self._path

    @property
    def entry_count(self) -> int:
        """Number of indexed skills."""
        return len(self._data.get("entries", {}))

    @property
    def entry_ids(self) -> list[str]:
        """List of all indexed skill IDs."""
        return list(self._data.get("entries", {}).keys())


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _compute_ahs_hash(ahs_spec: Any) -> str:
    """Compute a deterministic hash of an AHSSpec for change detection."""
    import hashlib

    # Hash the key fields that constitute the spec's identity
    raw = json.dumps({
        "identity": ahs_spec.identity.model_dump(),
        "interface": ahs_spec.interface.model_dump(),
        "intent": ahs_spec.intent.model_dump(),
    }, sort_keys=True, ensure_ascii=False)
    return "sha256:" + hashlib.sha256(raw.encode()).hexdigest()
