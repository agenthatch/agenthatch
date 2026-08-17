"""Regression tests for MemoryBrick search bugs found in v1.0.11 audit.

Covers:
- Bug #24: MemorySearch._escape_fts5_query inconsistent with KB store
  - Missing escapes for ^ and \\
  - Hyphens escaped (should be replaced with spaces)
  - Prefix wildcard only on last word (should be on every word)
  - AND join (should be OR for better recall)
- Bug #24b: MemorySearch._fallback_search doesn't escape LIKE wildcards
- Bug #25: MemorySearch._bm25_search score formula inverted
  - Uses 1/(1+|rank|) which gives HIGHER score to LESS relevant docs
  - KB store fixed this to |rank|/(1+|rank|) in v0.9.x; memory store missed
- Bug #26: _bm25_search catches Exception (too broad, masks programming errors)
- Bug #27: _bm25_search missing empty-query guard (LIKE fallback matches all)
- Bug #28: iter_session_entries / get_recent_session_entries skip entire
  file on any single corrupt JSON line (data loss)
- Bug #29: iter_session_entries / get_recent_session_entries / iter_knowledge_facts
  don't catch UnicodeDecodeError (subclass of ValueError, not OSError)
"""

from __future__ import annotations

import sqlite3
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Bug #24: _escape_fts5_query missing ^ and \\ escapes, wrong hyphen handling,
#           prefix wildcard only on last word, AND join
# ---------------------------------------------------------------------------

class TestBug24MemoryEscapeFts5Query:
    """Bug #24: MemorySearch._escape_fts5_query must match KB store v1.0.1
    pattern.  Previously:
      - Missing ^ and \\ escapes (silent query failures)
      - Hyphens escaped as literal (instead of replaced with spaces)
      - Prefix wildcard only on last word
      - Default space = AND join (too strict for RAG recall)
    """

    def _escape(self, query: str) -> str:
        from agenthatch_core.bricks.memory.search import MemorySearch
        return MemorySearch._escape_fts5_query(query)

    def test_hyphens_replaced_with_spaces(self) -> None:
        """Hyphenated query should be split, not escaped as literal."""
        result = self._escape("wind-rider")
        # Each word gets a prefix wildcard and OR join
        assert "wind*" in result, f"Expected wind* in {result!r}"
        assert "rider*" in result, f"Expected rider* in {result!r}"
        assert "OR" in result, f"Expected OR join in {result!r}"
        # Must NOT contain the literal escaped hyphen
        assert "\\-" not in result, (
            f"Hyphen should be replaced with space, not escaped; got {result!r}"
        )

    def test_caret_escaped(self) -> None:
        """``^`` (FTS5 column qualifier) must be escaped as literal."""
        result = self._escape("title^hello")
        assert "\\^" in result, (
            f"^ must be escaped to prevent column qualifier interpretation; "
            f"got {result!r}"
        )

    def test_backslash_escaped(self) -> None:
        """Backslash must be escaped (FTS5 escape prefix)."""
        result = self._escape("C:\\Users")
        assert "\\\\" in result, (
            f"Backslash must be escaped to prevent silent query truncation; "
            f"got {result!r}"
        )

    def test_prefix_wildcard_on_every_word(self) -> None:
        """Every word should get a ``*`` suffix, not just the last."""
        result = self._escape("fireball level 3 spell")
        words = result.split(" OR ")
        # All words should end with *
        for w in words:
            assert w.endswith("*"), (
                f"Every word should have prefix wildcard; word {w!r} doesn't"
            )

    def test_or_join_semantics(self) -> None:
        """Multiple words should be joined with OR for better recall."""
        result = self._escape("fireball spell")
        assert " OR " in result, (
            f"Multi-word queries should use OR join; got {result!r}"
        )

    def test_empty_query_returns_empty(self) -> None:
        """Empty query should return empty string (not '*' or 'OR ')."""
        assert self._escape("") == ""
        assert self._escape("   ") == ""

    def test_single_word_gets_prefix(self) -> None:
        """Single word should still get ``*`` prefix."""
        result = self._escape("fireball")
        assert result == "fireball*", f"Single word should get *; got {result!r}"

    def test_star_escaped(self) -> None:
        """Literal ``*`` in query should be escaped (not interpreted as prefix)."""
        result = self._escape("50*100")
        assert "\\*" in result, (
            f"Literal * must be escaped; got {result!r}"
        )


# ---------------------------------------------------------------------------
# Bug #24b: _fallback_search doesn't escape LIKE wildcards (%, _, \)
# ---------------------------------------------------------------------------

class TestBug24bMemoryFallbackSearchEscapesLike:
    """Bug #24b: MemorySearch._fallback_search must escape LIKE wildcards.

    Previously ``like_query = f"%{query}%"`` interpolated the raw user
    query.  Queries containing ``%`` or ``_`` would act as wildcards:
      - ``"50%"`` matches anything containing 50 followed by anything
      - ``"file_name"`` matches ``fileXname`` (underscore = any char)

    KB store got this fix in v1.0.1; memory store was missed.
    """

    def _build_search(self, tmp_path: Path):
        """Build a MemorySearch with one indexed entry containing % and _."""
        from agenthatch_core.bricks.memory.search import MemorySearch
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        # Insert a doc whose content contains % and _
        store.save_knowledge_fact("Discount code: 50%off for file_name.txt")
        search = MemorySearch(store)
        # _fallback_search queries memory_index table, which is created by
        # _ensure_index().  Without this, _fallback_search raises
        # "no such table: memory_index".
        search._ensure_index()
        return search

    def test_percent_in_query_does_not_act_as_wildcard(self, tmp_path: Path) -> None:
        """Query ``50%`` should match docs containing literal ``50%``, not
        any doc with ``50`` followed by anything."""
        search = self._build_search(tmp_path)
        # Add a doc that just has "50" without "%"
        search.store.save_knowledge_fact("50 spells total in the book")
        search._indexed = False
        search._ensure_index()
        # Query "50%" should match ONLY the doc with literal "50%off"
        # (not the "50 spells" doc which has no %)
        results = search._fallback_search("50%")
        contents = [r.content for r in results]
        assert any("50%off" in c for c in contents), (
            f"Should match doc with literal 50%off; got {contents}"
        )
        # If % were NOT escaped, the pattern would be %50%% which matches
        # anything containing 50 followed by anything — including "50 spells"
        assert not any("50 spells" in c for c in contents), (
            f"'50 spells' should NOT match query '50%' — if it does, the % "
            f"in the query is acting as a wildcard (escape missing). "
            f"Got: {contents}"
        )

    def test_underscore_in_query_does_not_act_as_wildcard(self, tmp_path: Path) -> None:
        """Query ``file_name`` should match ``file_name`` literally, not
        ``fileXname`` (underscore = any char in LIKE)."""
        search = self._build_search(tmp_path)
        # Add a doc with "fileXname" (where X is anything) that should NOT match
        search.store.save_knowledge_fact("Reference: fileXname pattern")
        search._indexed = False
        search._ensure_index()
        # Query for "file_name" — should match the doc with literal file_name.txt,
        # NOT the doc with "fileXname"
        results = search._fallback_search("file_name")
        contents = [r.content for r in results]
        assert any("file_name" in c for c in contents), (
            f"Should match doc with literal file_name; got {contents}"
        )
        # If _ were NOT escaped, LIKE would treat it as "any single char"
        # and match "fileXname" too
        assert not any("fileXname" in c for c in contents), (
            f"'fileXname' should NOT match query 'file_name' — if it does, "
            f"the _ in the query is acting as a wildcard (escape missing). "
            f"Got: {contents}"
        )


# ---------------------------------------------------------------------------
# Static source code guards (prevent regression during refactor)
# ---------------------------------------------------------------------------

class TestBug24SourceCodeGuards:
    """Static checks on the source file to prevent silent regression."""

    def test_escape_function_matches_kb_pattern(self) -> None:
        """Memory _escape_fts5_query should use the same regex as KB store."""
        src = Path(
            "agenthatch-core/src/agenthatch_core/bricks/memory/search.py"
        ).read_text()
        # Must escape ^ and \
        assert "^" in src and "\\\\" in src, (
            "Escape regex must include ^ and \\ — Bug 24 regression"
        )
        # Must replace hyphens with spaces (not escape them)
        assert '.replace("-", " ")' in src, (
            "Hyphens must be replaced with spaces, not escaped as literal"
        )
        # Must use OR join
        assert '" OR "' in src, (
            "Multi-word queries must be joined with OR for recall"
        )

    def test_fallback_search_escapes_like_wildcards(self) -> None:
        """_fallback_search must escape %, _, and \\ in LIKE queries."""
        src = Path(
            "agenthatch-core/src/agenthatch_core/bricks/memory/search.py"
        ).read_text()
        # Source file text contains literal \\% (two chars) because Python
        # source uses \\ to represent a single backslash in the string.
        assert r'replace("%", "\\%")' in src, (
            "_fallback_search must escape % in LIKE queries"
        )
        assert r'replace("_", "\\_")' in src, (
            "_fallback_search must escape _ in LIKE queries"
        )
        # Use raw string: the source file literally contains ESCAPE '\\'
        # (two backslash chars) because Python source uses \\ to escape.
        # A non-raw "ESCAPE '\\'" evaluates to "ESCAPE '\'" (one backslash)
        # and would never match the two-backslash source.
        assert r"ESCAPE '\\'" in src, (
            "_fallback_search must declare ESCAPE clause to enable backslash escaping"
        )


# ---------------------------------------------------------------------------
# Bug #25: _bm25_search score formula is INVERTED
# ---------------------------------------------------------------------------

class TestBug25Bm25ScoreInverted:
    """Bug #25: MemorySearch._bm25_search uses ``1/(1+|rank|)`` which gives
    a HIGHER score to LESS relevant documents.

    SQLite FTS5 ``bm25()`` returns negative values where MORE negative =
    MORE relevant.  The KB store fixed this in v0.9.x with the formula
    ``abs(rank) / (1 + abs(rank))`` (higher = more relevant).  The memory
    store was missed during the original refactor and still uses the
    inverted ``1 / (1 + abs(rank))`` (lower = more relevant).

    User-visible impact: ``search()`` returns results with the LEAST
    relevant document first, because ``_mmr_rerank`` sorts by score
    descending and the inverted scores put less-relevant docs on top.
    """

    def _build_search(self, tmp_path: Path):
        from agenthatch_core.bricks.memory.search import MemorySearch
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        # Doc A: high term frequency (more BM25-relevant for "fireball")
        store.save_knowledge_fact("fireball fireball fireball fireball fireball")
        # Doc B: low term frequency (less BM25-relevant)
        store.save_knowledge_fact("fireball")
        search = MemorySearch(store)
        search._ensure_index()
        return search

    def test_bm25_score_higher_for_more_relevant_doc(self, tmp_path: Path) -> None:
        """The doc with higher TF (more negative rank) must get a HIGHER
        score under the corrected formula.

        With the inverted ``1/(1+|rank|)`` formula, the high-TF doc gets
        a LOWER score (because |rank| is larger), so this assertion fails.
        """
        search = self._build_search(tmp_path)
        results = search._bm25_search("fireball")
        assert len(results) == 2, f"expected 2 results, got {len(results)}"

        by_content = {r.content: r.score for r in results}
        high_tf = "fireball fireball fireball fireball fireball"
        low_tf = "fireball"
        assert high_tf in by_content, f"high-TF doc missing: {by_content}"
        assert low_tf in by_content, f"low-TF doc missing: {by_content}"

        # The more relevant doc (higher TF → more negative rank → larger
        # |rank|) must get a HIGHER score under the corrected formula.
        assert by_content[high_tf] > by_content[low_tf], (
            f"More relevant doc should have HIGHER score. "
            f"high_tf={by_content[high_tf]!r}, low_tf={by_content[low_tf]!r}. "
            f"If high_tf < low_tf, the score formula is inverted (Bug #25)."
        )

    def test_search_returns_most_relevant_first(self, tmp_path: Path) -> None:
        """End-to-end: ``search()`` must return the most relevant doc first.

        This is the user-visible behavior.  With the inverted formula,
        ``_mmr_rerank`` sorts by the (inverted) score descending and puts
        the LEAST relevant doc first.
        """
        search = self._build_search(tmp_path)
        results = search.search("fireball", top_k=5)
        assert len(results) >= 1, "expected at least 1 result"
        first = results[0].content
        expected = "fireball fireball fireball fireball fireball"
        assert first == expected, (
            f"Most relevant doc should be returned first. "
            f"Expected {expected!r}, got {first!r}. "
            f"If the less-relevant 'fireball' doc is first, the BM25 "
            f"score formula is inverted (Bug #25)."
        )

    def test_score_formula_matches_kb_store(self) -> None:
        """Static guard: the score formula in source must use the
        ``abs(rank) / (1 + abs(rank))`` pattern, not the inverted
        ``1 / (1 + abs(rank))`` pattern.

        This prevents future refactors from reintroducing the bug.
        Uses AST to check only executable code, not comments/strings.
        """
        import ast

        src_path = Path(
            "agenthatch-core/src/agenthatch_core/bricks/memory/search.py"
        )
        src = src_path.read_text()
        tree = ast.parse(src)

        # Find the _bm25_search method and inspect its source
        bm25_methods = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "_bm25_search":
                bm25_methods.append(node)

        assert bm25_methods, "_bm25_search method not found in search.py"
        method_src = ast.get_source_segment(src, bm25_methods[0])

        # Correct formula pattern: magnitude = abs(rank); score = magnitude / (1 + magnitude)
        assert "magnitude = abs(rank)" in method_src, (
            "_bm25_search must use `magnitude = abs(rank)` pattern (Bug #25 fix). "
            f"Found in method body: {method_src!r}"
        )
        assert "score = magnitude / (1.0 + magnitude)" in method_src, (
            "_bm25_search must use `score = magnitude / (1.0 + magnitude)` "
            f"(Bug #25 fix). Found in method body: {method_src!r}"
        )
        # Must NOT contain the inverted formula in executable code
        assert "1.0 / (1.0 + abs(rank))" not in method_src, (
            "_bm25_search must NOT use inverted `1/(1+abs(rank))` formula (Bug #25). "
            f"Use `magnitude / (1.0 + magnitude)` instead. Method body: {method_src!r}"
        )


# ---------------------------------------------------------------------------
# Bug #27: _bm25_search missing empty-query guard
# ---------------------------------------------------------------------------

class TestBug27Bm25EmptyQueryGuard:
    """Bug #27: ``_bm25_search`` doesn't guard against empty queries.

    When the escaped query is empty (e.g. user passed only special chars
    that all get stripped), the FTS5 MATCH clause receives an empty
    string, which raises ``OperationalError: syntax error``.  This gets
    caught by the broad ``except Exception`` and falls through to
    ``_fallback_search``, which builds ``like_query = "%%"`` and matches
    EVERYTHING — returning 20 random docs to the user.

    KB store guards with ``if not safe_query.strip(): return []``;
    memory store should too.
    """

    def test_empty_query_returns_empty_list(self, tmp_path: Path) -> None:
        from agenthatch_core.bricks.memory.search import MemorySearch
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        store.save_knowledge_fact("some doc that should not be returned")
        search = MemorySearch(store)
        search._ensure_index()
        # Empty query — should return [], not all docs
        results = search._bm25_search("")
        assert results == [], (
            f"Empty query should return empty list, got {len(results)} results. "
            f"If non-empty, the empty-query guard (Bug #27) is missing."
        )

    def test_whitespace_only_query_returns_empty(self, tmp_path: Path) -> None:
        from agenthatch_core.bricks.memory.search import MemorySearch
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        store.save_knowledge_fact("some doc")
        search = MemorySearch(store)
        search._ensure_index()
        # Whitespace-only query — after _escape_fts5_query it becomes ""
        results = search._bm25_search("   ")
        assert results == [], (
            f"Whitespace-only query should return empty list, got {len(results)}."
        )


# ---------------------------------------------------------------------------
# Bug #28: iter_session_entries / get_recent_session_entries skip entire
# file on any single corrupt JSON line
# ---------------------------------------------------------------------------

class TestBug28IterSessionEntriesSkipsBadLineNotFile:
    """Bug #28: ``iter_session_entries`` and ``get_recent_session_entries``
    wrap the ENTIRE file read+parse loop in a single try/except.  If any
    one line has corrupt JSON, the whole file is skipped — losing all
    valid entries in that file.

    A single partial write (e.g. process killed mid-append) shouldn't
    cause the loss of all other valid entries in the same file.
    """

    def _write_session_file(self, sessions_dir: Path, filename: str, lines: list[str]) -> None:
        """Write a session file with the given raw lines (no validation)."""
        sessions_dir.mkdir(parents=True, exist_ok=True)
        (sessions_dir / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")

    def test_iter_session_entries_skips_bad_line_not_file(self, tmp_path: Path) -> None:
        """A corrupt line in the middle of a session file should be skipped,
        not the entire file."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        self._write_session_file(
            store._sessions_dir,
            "2026-01-01.jsonl",
            [
                '{"role": "user", "content": "valid entry 1", "timestamp": "2026-01-01T00:00:00Z"}',
                '{NOT VALID JSON',
                '{"role": "assistant", "content": "valid entry 2", "timestamp": "2026-01-01T00:01:00Z"}',
            ],
        )
        entries = store.iter_session_entries()
        contents = [e.get("content", "") for e in entries]
        # Both valid entries should be returned — only the bad line is skipped
        assert "valid entry 1" in contents, (
            f"valid entry 1 should survive bad-line skip; got {contents}"
        )
        assert "valid entry 2" in contents, (
            f"valid entry 2 should survive bad-line skip; got {contents}"
        )

    def test_get_recent_session_entries_skips_bad_line(self, tmp_path: Path) -> None:
        """Same bug in get_recent_session_entries."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        self._write_session_file(
            store._sessions_dir,
            "2026-01-01.jsonl",
            [
                '{"role": "user", "content": "recent valid", "timestamp": "2026-01-01T00:00:00Z"}',
                'CORRUPT LINE',
            ],
        )
        entries = store.get_recent_session_entries(limit=10)
        contents = [e.get("content", "") for e in entries]
        assert "recent valid" in contents, (
            f"recent valid entry should survive bad-line skip; got {contents}"
        )


# ---------------------------------------------------------------------------
# Bug #29: iter_session_entries / iter_knowledge_facts don't catch
# UnicodeDecodeError (subclass of ValueError, not OSError)
# ---------------------------------------------------------------------------

class TestBug29IterHandlesUnicodeDecodeError:
    """Bug #29: ``except (json.JSONDecodeError, OSError)`` doesn't catch
    ``UnicodeDecodeError`` because that's a subclass of ``ValueError``
    (via ``UnicodeError``), not ``OSError``.

    If a session file has non-UTF-8 bytes (e.g. truncated multi-byte
    sequence from a crash), ``read_text(encoding='utf-8')`` raises
    ``UnicodeDecodeError`` which propagates up and crashes the entire
    ``iter_session_entries`` call — taking down the agent's recall.
    """

    def _write_binary_file(self, path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def test_iter_session_entries_handles_bad_encoding(self, tmp_path: Path) -> None:
        """A session file with invalid UTF-8 bytes should be skipped,
        not crash the whole iter call."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        # Write a file with invalid UTF-8 (lone continuation byte 0x80)
        self._write_binary_file(
            store._sessions_dir / "2026-01-01.jsonl",
            b'{"role": "user", "content": "bad bytes: \x80 \xff"}\n',
        )
        # Should not raise — should skip the bad file and return []
        entries = store.iter_session_entries()
        assert isinstance(entries, list), (
            "iter_session_entries should return a list even on UnicodeDecodeError"
        )

    def test_iter_knowledge_facts_handles_bad_encoding(self, tmp_path: Path) -> None:
        """A knowledge file with invalid UTF-8 should be skipped, not crash."""
        from agenthatch_core.bricks.memory.store import MemoryStore

        store = MemoryStore(tmp_path)
        self._write_binary_file(
            store._knowledge_dir / "facts.jsonl",
            b'{"fact": "bad bytes: \x80 \xff", "timestamp": "2026-01-01"}\n',
        )
        entries = store.iter_knowledge_facts()
        assert isinstance(entries, list), (
            "iter_knowledge_facts should return a list even on UnicodeDecodeError"
        )


if __name__ == "__main__":
    sys.exit(0)
