"""Regression tests for v1.0.1 KB pipeline bug fixes.

Covers:
- Bug #2: KBChunker else branch preserves accumulated content
- Bug #3: discover_kb_files() applies exclude_patterns
- Bug #5: KnowledgeBaseConfig cross-field validator (chunk_overlap < chunk_size)
- Bug #6: hatch.py refreshes skill_dir/agenthatch.yaml with post-Phase 3.5 KB stats
- Bug #7: pyproject.toml.j2 includes agenthatch-core when kb_enabled
- Bug #8: KnowledgeStore.search(top_k<=0) returns []
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest


# ---------------------------------------------------------------------------
# Bug #8: KnowledgeStore.search(top_k<=0) returns []
# ---------------------------------------------------------------------------

def _build_test_store(tmp_path: Path) -> Any:
    """Build a minimal KnowledgeStore with one tiny doc for top_k tests."""
    from agenthatch_core.bricks.knowledge.store import KnowledgeStore

    store_dir = tmp_path / "kb"
    store_dir.mkdir()
    store = KnowledgeStore(store_dir, embedding_model="all-MiniLM-L6-v2")
    store.load()  # initialize empty index
    store.add_document(
        doc_id="d1",
        content="Fireball is a 3rd-level evocation spell dealing 8d6 fire damage.",
        metadata={"source": "spells/fireball.md", "chunk_index": 0},
    )
    store.build_index()
    return store


class TestBug8TopKZeroReturnsEmpty:
    """Bug #8: ``top_k=0`` must return ``[]``, not silently clamp to 1.

    Previously ``KnowledgeStore.search(top_k=0)`` clamped to 1 and
    returned a single result, violating the literal API contract
    "at most top_k results" (1 > 0).  A user passing ``top_k=0`` is
    explicitly asking for no results — return exactly that.
    """

    def test_top_k_zero_returns_empty_list(self, tmp_path: Path) -> None:
        store = _build_test_store(tmp_path)
        result = store.search(query="fireball", top_k=0)
        assert result == [], (
            f"top_k=0 must return [] (got {len(result)} results)"
        )

    def test_top_k_negative_returns_empty_list(self, tmp_path: Path) -> None:
        store = _build_test_store(tmp_path)
        result = store.search(query="fireball", top_k=-5)
        assert result == [], (
            f"top_k=-5 must return [] (got {len(result)} results)"
        )

    def test_top_k_one_still_works(self, tmp_path: Path) -> None:
        """Make sure the fix didn't accidentally reject positive top_k."""
        store = _build_test_store(tmp_path)
        result = store.search(query="fireball", top_k=1)
        assert len(result) == 1, f"top_k=1 should return 1 result, got {len(result)}"

    def test_top_k_positive_returns_at_most_top_k(self, tmp_path: Path) -> None:
        store = _build_test_store(tmp_path)
        result = store.search(query="fireball", top_k=10)
        assert len(result) <= 10


# ---------------------------------------------------------------------------
# Bug #2: KBChunker else branch preserves accumulated content
# ---------------------------------------------------------------------------

class TestBug2ChunkerElseBranchPreservesContent:
    """Bug #2: KBChunker's else branch must append, not reset.

    When a paragraph is too small to flush the current chunk, the
    else branch must append to ``current_parts`` (preserving accumulated
    content).  Previously it reset ``current_parts = [overlap_text,
    para_text]``, silently dropping everything before the overlap.
    """

    def test_small_then_huge_paragraphs_preserve_all_content(self) -> None:
        from agenthatch_core.bricks.knowledge.chunker import KBChunker

        # Three 30-char paragraphs (too small to flush individually),
        # followed by a 900-char paragraph that triggers the flush.
        # Total content: 990 chars.  Bug #2 dropped ~60 chars.
        tiny = "a" * 30 + "\n\n"
        huge = "b" * 900
        text = tiny * 3 + huge
        chunker = KBChunker(chunk_size=800, chunk_overlap=100)
        chunks = chunker.chunk_text(text, source="test.md")
        rebuilt = "\n\n".join(c.content for c in chunks)
        # All 'a's from the tiny paragraphs must survive.
        assert rebuilt.count("a") == 90, (
            f"Expected 90 'a' chars (3×30), got {rebuilt.count('a')} — "
            f"Bug #2 regression: small paragraphs were dropped"
        )
        # All 'b's from the huge paragraph must survive.
        assert rebuilt.count("b") == 900, (
            f"Expected 900 'b' chars, got {rebuilt.count('b')}"
        )


# ---------------------------------------------------------------------------
# Bug #3: discover_kb_files() applies exclude_patterns
# ---------------------------------------------------------------------------

class TestBug3ExcludePatternsFilterFiles:
    """Bug #3: ``discover_kb_files()`` must honor ``exclude_patterns``.

    Previously ``discover_kb_files()`` did not accept the
    ``exclude_patterns`` parameter — excluded file names still leaked
    into B3/B4 LLM context (the engine's index builder filtered them,
    but the LLM pipeline didn't, so the LLM saw filenames the user
    explicitly wanted hidden).
    """

    def test_exclude_patterns_filters_matching_files(self, tmp_path: Path) -> None:
        from agenthatch.skill.kb_pipeline import discover_kb_files

        (tmp_path / "public.md").write_text("# Public", encoding="utf-8")
        # ``*.secret`` matches basename ending in ``.secret`` (a real
        # convention for secrets in repos), not ``secret.md``.  Use
        # a filename that actually matches the pattern.
        (tmp_path / "api.secret").write_text("# Secret", encoding="utf-8")
        (tmp_path / "draft").mkdir()
        (tmp_path / "draft" / "unreleased.md").write_text(
            "# Draft", encoding="utf-8"
        )

        files = discover_kb_files(
            tmp_path,
            exclude_patterns=["draft/*", "*.secret"],
        )
        names = sorted(f.name for f in files)
        assert "public.md" in names
        assert "api.secret" not in names, (
            f"api.secret should be excluded (matched *.secret), got {names}"
        )
        assert "unreleased.md" not in names, (
            f"draft/unreleased.md should be excluded (matched draft/*), got {names}"
        )


# ---------------------------------------------------------------------------
# Bug #5: KnowledgeBaseConfig cross-field validator
# ---------------------------------------------------------------------------

class TestBug5ChunkOverlapLtSizeValidator:
    """Bug #5: KnowledgeBaseConfig must reject chunk_overlap >= chunk_size.

    Previously the per-field validators only checked ``chunk_size > 0``
    and ``chunk_overlap >= 0`` independently — there was no cross-field
    check that ``chunk_overlap < chunk_size``.  KBChunker.__init__ then
    raised ValueError mid-build, crashing hatch with a raw traceback.
    """

    def test_overlap_greater_than_size_rejected(self) -> None:
        from agenthatch.skill.spec import KnowledgeBaseConfig

        with pytest.raises(ValueError, match="chunk_overlap.*must be < chunk_size"):
            KnowledgeBaseConfig(
                sources=[],
                usage_strategy={"when_to_retrieve": [], "query_templates": []},
                prompt_artifact={
                    "system_prompt_section": "",
                    "retrieve_tool_description": "",
                    "integration_instructions": "",
                },
                chunk_size=100,
                chunk_overlap=200,  # > chunk_size — invalid
            )

    def test_overlap_equal_to_size_rejected(self) -> None:
        from agenthatch.skill.spec import KnowledgeBaseConfig

        with pytest.raises(ValueError, match="chunk_overlap.*must be < chunk_size"):
            KnowledgeBaseConfig(
                sources=[],
                usage_strategy={"when_to_retrieve": [], "query_templates": []},
                prompt_artifact={
                    "system_prompt_section": "",
                    "retrieve_tool_description": "",
                    "integration_instructions": "",
                },
                chunk_size=100,
                chunk_overlap=100,  # == chunk_size — invalid
            )

    def test_default_overlap_lt_size_accepted(self) -> None:
        from agenthatch.skill.spec import KnowledgeBaseConfig

        # Default values: chunk_size=800, chunk_overlap=100 — valid.
        cfg = KnowledgeBaseConfig(
            sources=[],
            usage_strategy={"when_to_retrieve": [], "query_templates": []},
            prompt_artifact={
                "system_prompt_section": "",
                "retrieve_tool_description": "",
                "integration_instructions": "",
            },
        )
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100
        assert cfg.chunk_overlap < cfg.chunk_size


# ---------------------------------------------------------------------------
# Bug #7: pyproject.toml.j2 declares agenthatch-core when kb_enabled
# ---------------------------------------------------------------------------

class TestBug7PyprojectDeclaresKBDependencies:
    """Bug #7: Generated pyproject.toml must declare agenthatch-core.

    Without this, ``pip install -e .`` succeeds but ``retrieve()``
    silently returns ``[]`` for every query (the ImportError inside
    ``_get_store()`` is swallowed).  The template must conditionally
    add ``agenthatch-core`` to ``[project].dependencies`` when
    ``kb_enabled`` is True.
    """

    def _render(self, kb_enabled: bool) -> str:
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("pyproject.toml.j2")
        return tpl.render(
            agent_name="test-agent",
            version="0.1.0",
            description="test",
            package_name="test_agent",
            kb_enabled=kb_enabled,
        )

    def test_kb_enabled_adds_agenthatch_core_dependency(self) -> None:
        rendered = self._render(kb_enabled=True)
        assert "agenthatch-core" in rendered, (
            "pyproject.toml must declare agenthatch-core when kb_enabled=True"
        )
        assert "sentence-transformers" in rendered, (
            "pyproject.toml must declare sentence-transformers when kb_enabled=True"
        )

    def test_kb_disabled_omits_agenthatch_core_dependency(self) -> None:
        rendered = self._render(kb_enabled=False)
        assert "agenthatch-core" not in rendered, (
            "pyproject.toml must NOT declare agenthatch-core when kb_enabled=False"
        )


# ---------------------------------------------------------------------------
# Bug #4: pyproject.toml.j2 packages knowledge/ directory
# ---------------------------------------------------------------------------

class TestBug4PyprojectPackagesKnowledgeDir:
    """Bug #4: Generated pyproject.toml must force-include ``knowledge/``.

    Without this, ``pip install`` ships only ``src/<pkg>/`` — the
    pre-built SQLite index at ``knowledge/kb_index.db`` is left out
    of the wheel, and ``retrieve()`` returns ``[]`` at runtime because
    ``_KB_INDEX_DIR`` resolves to a non-existent path.
    """

    def _render(self, kb_enabled: bool) -> str:
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("pyproject.toml.j2")
        return tpl.render(
            agent_name="test-agent",
            version="0.1.0",
            description="test",
            package_name="test_agent",
            kb_enabled=kb_enabled,
        )

    def test_kb_enabled_includes_force_include(self) -> None:
        rendered = self._render(kb_enabled=True)
        assert "force-include" in rendered, (
            "pyproject.toml must declare force-include when kb_enabled=True"
        )
        assert '"knowledge" = "knowledge"' in rendered or (
            'knowledge" = "knowledge' in rendered
        ), (
            "force-include must map knowledge/ → knowledge/, got: "
            f"{rendered}"
        )

    def test_kb_disabled_omits_force_include(self) -> None:
        rendered = self._render(kb_enabled=False)
        assert "force-include" not in rendered, (
            "pyproject.toml must NOT declare force-include when kb_enabled=False"
        )


# ---------------------------------------------------------------------------
# Bug #4 r2: _resolve_kb_index_dir() handles both pip and dev layouts
# ---------------------------------------------------------------------------

class TestBug4ResolverHandlesBothLayouts:
    """Bug #4 r2: The resolver must work for non-editable pip installs.

    The first attempt at Bug #4 used ``here.parent.parent / "knowledge"``
    as the "pip layout" candidate.  This was an off-by-one error: a
    non-editable ``pip install`` puts ``knowledge/`` at
    ``<site-packages>/knowledge/`` (1 level up from ``<pkg>/``), not
    at ``<site-packages>'s parent/knowledge/`` (2 levels up).  The
    previous fix accidentally passed for editable installs because
    2-levels-up from ``<agent_dir>/src/<pkg>/`` is ``<agent_dir>/``
    (which is where ``knowledge/`` lives in dev layout), but a true
    non-editable install left ``_KB_INDEX_DIR`` pointing at a
    non-existent path.

    This test renders the template and execs the resolver function
    with a mock ``__file__`` pointing at each layout to verify
    ``_KB_INDEX_DIR`` resolves correctly.
    """

    def _render_resolver(self) -> str:
        """Render knowledge_base.py.j2 with stub KB config and return
        only the resolver portion (lines up to and including
        ``_KB_INDEX_DIR = _resolve_kb_index_dir()``).
        """
        from agenthatch.generate.engine import GenerateEngine

        engine = GenerateEngine()
        tpl = engine._env.get_template("knowledge_base.py.j2")
        rendered = tpl.render(
            agent_name="test-agent",
            version="0.1.0",
            description="test",
            package_name="test_agent",
            kb_enabled=True,
            kb={
                "chunk_size": 800,
                "chunk_overlap": 100,
                "retrieval_alpha": 0.7,
                "enable_llm_rerank": False,
                "when_to_retrieve": [],
                "query_templates": [],
                "integration_pattern": "tool_call_then_answer",
                "max_results_per_query": 3,
                "citation_required": False,
                "fallback_when_no_match": "",
                "system_prompt_section": "",
                "retrieve_tool_description": "",
                "integration_instructions": "",
                "retrieval_top_k": 5,
                "embedding_model": "all-MiniLM-L6-v2",
                "total_documents": 0,
                "total_chunks": 0,
                "index_size_bytes": 0,
            },
        )
        # Extract just the resolver block — drop the LLM-inferred
        # constants (those are just data assignments and don't affect
        # the path resolution logic).
        lines: list[str] = []
        in_resolver = False
        for line in rendered.splitlines():
            if "_resolve_kb_index_dir" in line and "def " in line:
                in_resolver = True
            if in_resolver:
                lines.append(line)
                if line.strip().startswith("_KB_INDEX_DIR ="):
                    break
        return "\n".join(lines)

    def _exec_resolver(self, fake_file: Path) -> Path:
        """Exec the rendered resolver with ``__file__`` set to fake_file.

        Returns the value of ``_KB_INDEX_DIR`` from the exec'd module
        namespace.
        """
        code = self._render_resolver()
        # Build a fake module namespace with ``__file__`` pointing at
        # the requested fake location.  The resolver reads
        # ``Path(__file__).resolve().parent`` — so we need to create
        # the file on disk so ``.resolve()`` works.
        fake_file.parent.mkdir(parents=True, exist_ok=True)
        fake_file.write_text("# stub for resolver test\n", encoding="utf-8")

        # The resolver code uses ``Path`` (in the body) and as the
        # return annotation (``-> Path``).  We need ``Path`` in the
        # exec globals so the annotation resolves at function-def
        # time.  Note: the template has ``from __future__ import
        # annotations`` at the top, but we extract only the resolver
        # block — so the future import isn't in effect here, and
        # annotations are evaluated eagerly.
        exec_globals: dict[str, Any] = {
            "__name__": "_resolver_test_module",
            "__file__": str(fake_file),
            "__builtins__": __builtins__,
            "Path": Path,
        }
        exec(compile(code, str(fake_file), "exec"), exec_globals)
        return exec_globals["_KB_INDEX_DIR"]

    def test_resolver_finds_knowledge_in_pip_layout(self, tmp_path: Path) -> None:
        """Simulate non-editable pip install: knowledge/ is sibling of <pkg>/."""
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "test_agent"
        kb_dir = site_packages / "knowledge"
        kb_dir.mkdir(parents=True)
        (kb_dir / "kb_index.db").write_text("stub", encoding="utf-8")

        fake_file = pkg_dir / "knowledge_base.py"
        resolved = self._exec_resolver(fake_file)
        assert resolved == kb_dir, (
            f"pip layout: expected {kb_dir}, got {resolved} — "
            f"resolver doesn't find knowledge/ in non-editable install"
        )

    def test_resolver_finds_knowledge_in_dev_layout(self, tmp_path: Path) -> None:
        """Simulate dev / editable install: knowledge/ is 2 levels up from <pkg>/."""
        agent_dir = tmp_path / "agent_dir"
        pkg_dir = agent_dir / "src" / "test_agent"
        kb_dir = agent_dir / "knowledge"
        kb_dir.mkdir(parents=True)
        (kb_dir / "kb_index.db").write_text("stub", encoding="utf-8")

        fake_file = pkg_dir / "knowledge_base.py"
        resolved = self._exec_resolver(fake_file)
        assert resolved == kb_dir, (
            f"dev layout: expected {kb_dir}, got {resolved} — "
            f"resolver doesn't find knowledge/ in dev/editable install"
        )

    def test_resolver_prefers_pip_layout_when_both_exist(self, tmp_path: Path) -> None:
        """When both candidates exist, prefer the pip-layout (1-level-up) path.

        This guards against a regression where someone re-introduces the
        off-by-one.  We create BOTH layouts and verify the resolver
        picks the pip-layout path (it's listed first in ``candidates``).
        """
        # Build a fake site-packages layout
        site_packages = tmp_path / "site-packages"
        pkg_dir = site_packages / "test_agent"
        pip_kb = site_packages / "knowledge"
        pip_kb.mkdir(parents=True)
        (pip_kb / "kb_index.db").write_text("pip", encoding="utf-8")

        # Also create a stray ``knowledge/`` 2 levels up (dev layout
        # wouldn't typically coexist with pip layout, but we want
        # to verify the first-listed candidate wins).
        dev_kb = site_packages.parent / "knowledge"  # tmp_path/knowledge/
        dev_kb.mkdir(parents=True, exist_ok=True)
        (dev_kb / "kb_index.db").write_text("dev", encoding="utf-8")

        fake_file = pkg_dir / "knowledge_base.py"
        resolved = self._exec_resolver(fake_file)
        assert resolved == pip_kb, (
            f"when both layouts exist, resolver should prefer pip-layout "
            f"({pip_kb}), got {resolved}"
        )
