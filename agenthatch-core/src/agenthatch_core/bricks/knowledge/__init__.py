"""KnowledgeBaseBrick — RAG-native runtime brick for skillagent (v1.0.0).

Loads a pre-built vector + keyword index at agent startup and exposes
the ``retrieve`` tool to the LLM via CapBus.  The index is built during
hatch Phase 3.5 and persisted as a SQLite database.

Design inspired by OpenClaw's knowledge-base skill pattern:
  - Local-first (SQLite + optional embeddings, no cloud)
  - Small chunks (500-1000 chars, paragraph-boundary)
  - Hybrid retrieval (BM25 + embedding fusion)
  - Optional LLM re-rank for precision

Unlike MemoryBrick (which stores the agent's own memories), KnowledgeBaseBrick
holds external authoritative knowledge provided by the user at hatch time.
"""

from __future__ import annotations

from .store import KBDocument, KBSearchResult, KnowledgeStore
from .tools import RetrieveTool, retrieve_tool

__all__ = [
    "KBDocument",
    "KBSearchResult",
    "KnowledgeStore",
    "RetrieveTool",
    "retrieve_tool",
]
