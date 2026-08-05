# CHANGELOG

All notable changes to agenthatch will be documented in this file.

---

## [Unreleased]

No unreleased changes.

---

## [v1.0.7] — 2026-07-20

### Added

- **Exception-swallow antipattern detection** — The post-generation review now flags generated tool functions that catch `Exception` (or `BaseException`, or bare `except:`) and silently swallow with `pass` / `return None` / `return ""` / `return []` / `return 0` / `return False`. WARNING-only by design: an earlier attempt let B4 rewrite these automatically, but the LLM kept either deleting legitimate fallbacks (breaking resilience) or narrowing to a specific exception class that missed the actual failure mode. The smell is surfaced for the user to tighten by hand.

### Fixed

- **MCP tool name parsing in template** — `tools.py.j2` was splitting MCP tool names on the wrong delimiter, producing garbled server/tool identifiers in the generated CapBus registration.
- **Postgen review local-name detection** — `_detect_undefined_variables` now recognizes `AugAssign`, `MatchAs`, `MatchStar`, and `MatchMapping` nodes so variables bound by `+=` / `case x:` / `case *rest:` / `case {**rest}:` are no longer false-positive undefined.

---

## [v1.0.6] — 2026-07-20

### Fixed

- **`store.close()` crash when `KnowledgeStore` init fails** — `KnowledgeStore()` was instantiated before the `try` block, so if init itself failed (bad model path, disk full), the `finally` block called `store.close()` on an unbound variable. Moved instantiation inside `try` with a `None` guard on close.
- **Missing `retrieve` import in generated KB agents** — `agent.py.j2` with `kb_enabled` was missing `from .knowledge_base import retrieve`, so the first `retrieve()` call raised `NameError`. Template now emits the import.
- **Spinner freeze on E harness failure** — When E harness failed after retries, the progress spinner kept spinning because the harness status was never marked complete. `SchemaValidationError` is now caught, harness completion is displayed, and a user-friendly error panel replaces the raw traceback.
- **Token savings estimates outdated** — `validate.py` still quoted 5-harness token estimates (~8000) after the 6-harness migration. Updated to ~18000 to reflect the F harness addition.

---

## [v1.0.5] — 2026-07-20

### Changed

- **Confidence retry penalty** — Each harness retry now deducts 5% from the confidence score, floored at 0.75. Prevents a harness that succeeds on retry 3 from reporting the same confidence as one that succeeded on the first try.
- **E harness merges F's MCP servers** — Harness E now pulls MCP server definitions from F's output into the assembled interface, so generated agents ship with MCP tools wired in without a separate assembly step.
- **Harness B simplified frontmatter passing** — Frontmatter is now passed to B as-is rather than through an intermediate dict reshape that dropped unknown keys.

### Fixed

- **JSON Schema enum and constraints not mapped** — `compile_output_schema` now maps `enum` to `Literal`, numeric `minimum`/`maximum` to `ge`/`le`, string `minLength`/`maxLength`/`pattern` to `StringConstraints`, and array `minItems`/`maxItems` to `min_length`/`max_length`. Previously these constraints were silently dropped, so generated tool signatures accepted any value.
- **Internal validation errors leaked to users** — `_try_validate` was surfacing raw `ValidationError` messages (including internal field paths) in user-facing output. Now logs the full exception via `logger.error(exc_info=True)` and returns a generic error message.

---

## [v1.0.4] — 2026-07-19

### Fixed

- **Bug #10: `retrieve(top_k=N)` silently clamped by `RETRIEVAL_TOP_K`** — The generated `knowledge_base.py.j2` template called `store.search(top_k=min(top_k, RETRIEVAL_TOP_K), ...)`, clamping the caller's explicit `top_k` to the frontmatter-configured `RETRIEVAL_TOP_K` (default 5). An LLM calling `retrieve(top_k=20)` got 5 results with no warning and no way to distinguish "clamped" from "index only has 5 matches". Fixed by removing the `min()` clamp; `top_k` passes through to `store.search()` directly. `RETRIEVAL_TOP_K` remains as a build-time default (via `MAX_RESULTS_PER_QUERY` as the `retrieve()` signature default) but no longer caps runtime calls. 5 regression tests added in `TestBug10RetrieveTopKNotSilentlyClamped`, including 1 static source guard.
- **Bug #9: `_fuse_results` leaks zero-score results at alpha extremes** — `KnowledgeStore._fuse_results` uses `final_score = alpha * keyword + (1-alpha) * embedding`. At `alpha=1.0` (pure keyword mode), embedding-only documents still entered the fused dict with `(1-1.0)*score = 0`, occupying top-k slots with zero scores; `alpha=0.0` symmetrically leaked BM25-only zeros. Fixed with early-return at `alpha >= 1.0` (BM25-only) and `alpha <= 0.0` (embedding-only), falling back to the other side if the "pure" side is empty. Mixed alpha (0 < alpha < 1) keeps the v1.0.0 fusion logic. 8 regression tests added in `TestBug9FuseResultsNoLeakAtAlphaExtremes`.

### Added

- **KB regression test expansion (Bug #11-#16)** — 19 new tests covering previously untested KB core paths: `_escape_fts5_query` FTS5 special-char escaping, `_fuse_results` edge cases, `get_stats()` document counts, `KBChunker` edge cases (empty text / binary detection / Unicode errors / overlap=0), `_split_paragraphs` heading tracking, `_fallback_search` LIKE wildcard escaping. `tests/test_kb_regressions.py` grew from 29 to 48 tests.
- **Generated code static guard tests (R4-V23/V22/V16)** — New `tests/test_generated_code_regressions.py` (12 tests): prevents template `return (yield from` regression (R4-V23), `kb_max_text` typo (R4-V22), `sys.modules[spec.name]` omission (R4-V16), and `python_escape` null-byte / control-char / triple-quote handling.
- **MCP/LLM regression tests** — New `tests/test_mcp_regressions.py` (3 tests) guards `split("__", 2)` for three-segment MCP tool names. `tests/test_llm_regression.py` gains 2 tests guarding `ThinkingDelta` deferred import and `reasoning_tokens` getattr.

---

## [v1.0.3] — 2026-07-18

### Fixed

- **Bug #6: `agenthatch.yaml` in skill directory reports `total_chunks=0` after hatch** — The YAML was written before the KB index build, so the persisted file always showed `total_chunks=0` and `index_size_bytes=0`. `hatch.py` now refreshes `skill_dir/agenthatch.yaml` after the KB index build with accurate stats.
- **Bug #7: Generated agent's `pyproject.toml` missing `agenthatch-core` dependency** — KB-enabled agents silently failed on `retrieve()` because `agenthatch-core` (which provides `KnowledgeStore`) was not in the dependency list. `pyproject.toml.j2` now adds `agenthatch-core` and `sentence-transformers` when `kb_enabled`, and force-includes the `knowledge/` directory in the wheel.
- **Bug #8: `retrieve(top_k=0)` returns 1 result** — `KnowledgeStore.search` clamped `top_k <= 0` to 1 instead of returning an empty list. An explicit `top_k=0` (user asking for zero results) returned one result. Fixed to return `[]` for non-positive `top_k`.
- **KB index path resolution fails on pip install layout** — `knowledge_base.py.j2` hardcoded the KB index directory relative to the agent module, which broke under pip-installed layout (where `knowledge/` lives one level up from the dev layout). Added `_resolve_kb_index_dir()` with candidate-path fallback supporting both dev and pip layouts.

---

## [v1.0.2] — 2026-07-17

### Fixed

- **Bug #2: `KBChunker` else branch drops accumulated content** — When a chunk exceeded `chunk_size`, the else branch reset `current_parts` to a new list instead of appending the overflow paragraph, silently losing accumulated text. Fixed to append `para_text` to `current_parts` before starting a new chunk.
- **Bug #3: `discover_kb_files()` ignores `exclude_patterns`** — The exclude parameter was accepted but never applied. Files matching exclude patterns (e.g. `*.tmp`, `drafts/`) were indexed alongside real content. Fixed to filter files through `exclude_patterns` before returning.
- **`KBChunker` crashes when `chunk_overlap >= chunk_size`** — `KnowledgeBaseConfig` now validates `chunk_overlap < chunk_size` at construction time, preventing `KBChunker` initialization from raising `ValueError` mid-hatch.
- **Anti-narration patterns** — Expanded the meta-narration pattern list and tightened the sentence-boundary expansion logic to catch LLM variants like "already answered above" without deleting legitimate closing sentences.
- **CI mypy: `model_validator` return type** — `def _validate_chunk_overlap_lt_size(self) -> "KnowledgeBaseConfig"` used a string forward reference that confused mypy under `--strict`. Changed to a direct type reference.

---

## [v1.0.1] — 2026-07-16

### Fixed

- **R4-V23: `chat_stream()` bare `yield from` discards subgenerator return value** — Generated agents used `yield from super().chat_stream(user_input)`, which drops the subgenerator's `return` value (the `yield from` expression evaluates to `None` unless written as `return (yield from ...)`). `ConversationLoop.stream()` returned a complete `final_text` (e.g. 466 chars), but callers got an empty string. The text had streamed to the terminal, but the programmatic API returned nothing. Fixed to `return (yield from super().chat_stream(user_input))`, and patched `agent.py.j2` to prevent regression.
- **R4-V22: `kb_max_text` typo in `chat_stream()` path** — `agent.py` had `kb_max_text = 1` in the streaming path with a comment saying "Same KB auto-continuation cap as chat()", but `chat()` actually used `0`. That single digit meant `self._max_consecutive_text_only == 0` was never true on the streaming path, so `_strip_trailing_meta_narration` never ran and meta-narration leaked to the user. Fixed to `0`, matching `chat()`.
- **R4-V22: `_strip_trailing_meta_narration` old strategy deleted body text** — The old "truncate from earliest match to end" strategy deleted everything after the meta-narration (including legitimate closing invitations like "If you'd like to explore..."), which tripped the 40% safety guard and let the meta-narration through anyway. Rewritten to "delete the containing sentence": find all matches, expand to sentence boundaries, merge overlapping ranges, delete only those sentences. Safety threshold adjusted from 40% to 50%.
- **R4-V22: meta-narration pattern list expanded** — Added LLM variants ("already answered", "answered above", "detailed earlier", etc.) and made the "detailed" pattern optional so shorter forms like "already answered" also match.
- **R4-V21: meta-narration residue in text stream** — KB agents often append meta-commentary ("fully answered... no remaining steps") before `task_complete`, despite B4 (e) explicitly forbidding it. Added `_strip_trailing_meta_narration()` to match meta-narration patterns in the last 600 characters and delete the containing sentence.
- **R4-V20: `task_complete` re-yields meta-summary** — When `task_complete` was called after text had already streamed (`has_yielded_text=True`), it still yielded the `summary` argument (usually a meta-summary like "Answered..."), so users saw the answer followed by a redundant summary. Fixed: when `has_yielded_text=True`, use `accumulated_text` as `final_text` and don't yield the summary.
- **R4-V16: KB package name resolution via MRO causes import failure** — `type(self).__module__` sometimes resolved through MRO to the base class `agenthatch_core.agent` (whose `__package__` is "agenthatch_core"), so `importlib.import_module(f"{pkg}.knowledge_base")` failed with `No module named 'agenthatch_core.knowledge_base'`. Fixed to derive the package name directly from `__module__`, and register the agent module in `sys.modules[spec.name]` before `exec_module` in `run.py`.

### Changed

- **KB agent auto-continuation suppression** — Introduced `max_consecutive_text_only` and `nudge_grace` parameters (R4-V17). KB agents pass `0` so the loop returns after the first text-only response, avoiding auto-continuation producing duplicate answers and meta-summaries.

---

## [v1.0.0] — 2026-07-14

### Added

- **KnowledgeBaseBrick (RAG retrieval)** — An engineering knowledge base for agents, distinct from the skill's internal `references/` co-located knowledge. Users specify a KB path as the second CLI argument; the hatch pipeline builds a vector index at hatch time and the `retrieve` tool searches it at runtime.
  - **Phase B (compile-time integration)**: KB inference pipeline (B2 detection -> B3 usage strategy -> B4 prompt generation). `_build_knowledge_index()` runs after `_prepare_output_dir` and before template rendering.
  - **Phase C (runtime assembly)**: `RetrieveTool` registers on CapBus; `AHCoreAgent` KB assembly block; `ContextManager` injects the KB system prompt.
  - **`knowledge_base.py.j2` template**: generates a runtime `retrieve()` function with LLM-inferred `WHEN_TO_RETRIEVE`, `QUERY_TEMPLATES`, and `SYSTEM_PROMPT_SECTION`.
  - **SQLite FTS5 index + BM25 scoring**: FTS5 indexing replaces `-` with spaces to avoid NOT operator parsing; BM25 scoring uses `abs(rank)/(1+abs(rank))` to prioritize relevant documents.
- **HatchReport confidence source unified** — Hatch Summary confidence values now use E harness cross-evaluation scores, with self-assessment as fallback. Fixes the Confidence panel (1.00) vs Hatch Summary (0.50) mismatch.
- **Phase 3/3 title** — Normal (non-dry-run) hatch runs now display `Phase 3/3 Agent Generation` in the console.

### Fixed

- **`--force` deletes KB index** — The `--force` flag was deleting KB index files when overwriting the output directory. Fixed: `--force` no longer clears the KB index.
- **SQLite cross-thread errors** — SQLite connections now use thread-local storage to prevent cross-thread access errors.
- **BM25 score inversion** — BM25 scores were not taking absolute values, so negative ranks produced inverted scores. Fixed to `abs(rank)/(1+abs(rank))`.
- **FTS5 hyphen parsing** — FTS5 parsed hyphens as NOT operators, causing query failures. Fixed by replacing `-` with spaces at index time.
- **B2 detector false positives** — B2 detector matched directory names, misclassifying non-KB skills as KB skills. Fixed to only recognize KB-specific vocabulary, not directory names.

---

## [v0.9.23] — 2026-07-04

### Roadmap

- **Phase 1 (Quality & Observability) concluded.** The post-generation self-review loop shipped in v0.9.22 was the final planned Phase 1 deliverable. Remaining observability hooks (per-round token counts, iteration traces, repair diffs) are deferred — open a Discussion if a specific signal is needed.
- **Phase 2 (Intelligence) is the next active focus.** Knowledge-backed agents (RAG-native skillagent) — agents that ship with their own vector index and retrieve relevant references per query — is now the primary roadmap target. See `ROADMAP.md`.

### Added

- **PlanLayer state machine test suite** — 82 tests covering all 6 states (STARTING/PLANNING/EXECUTING/VERIFYING/REPLANNING/DONE), state transitions, failure keyword detection, MAX_CONSECUTIVE_FAILURES threshold, VERIFY_EVERY_N_STEPS checkpoint, nag_limit (plan_guided=4/conversation=2), to_context_text rendering (☐▶✓✗), serialization
- **SkillhouseIndex test suite** — 54 tests covering hybrid search (BM25 α=0.7 + embedding), lazy init, embedding degradation, topological sort (Kahn's algorithm + circular dependency), atomic save, _compute_ahs_hash, find_provider, CRUD operations
- **Engine orchestrator test suite** — 36 tests covering HARNESS_CONFIG (temperatures, thinking, reasons), HARNESS_LABELS mapping, MODEL_TIER_MAP (skill type → model tier, pure_instruction skips D), should_skip_reflection confidence thresholds (A/F ≥ 0.9, E ≥ 0.95, never skip with errors)
- **Post-generation review design document** — `docs/agenthatch-v0.9.22-postgen-review-design.md`. Designs Phase 3.5 post-gen review: inspection checks, tool self-test strategy, iteration loop (max 3 rounds), quality gate definition. Implementation deferred — stub frequency 0% in real hatch measurement.
- **Timeout mechanism evaluation document** — `docs/v0.9.22-timeout-evaluation.md`. Evaluates three alternatives (multiprocessing, asyncio, keep current) for `_route_with_timeout()`. Recommendation: keep current ThreadPoolExecutor + document limitation for v0.9.22; re-evaluate asyncio for v2.0.
- **Post-generation code inspection module** — `src/agenthatch/skill/postgen_review.py`. Detects undefined variables, None attribute access, semantic stubs in generated tools.py. Reuses `GenerateEngine._validate_generated_python()` (AST syntax + JS artifact detection) and `GenerateEngine._check_tool_stubs()` (literal stub detection), then adds three new checks: undefined variable detection (catches NameError bugs), None attribute access detection (catches AttributeError bugs), semantic stub detection (catches placeholder/template returns).
- **Tool self-test in post-generation review** — Calls each tool with default parameters, captures NameError/TypeError/AttributeError. Sandbox-isolated (subprocess), 10s timeout. Side-effect detection skips tools with subprocess/network/file-IO calls.
- **Autonomous quality-gate iteration loop** — `iterate_until_gate()` runs inspect → test → fix → re-inspect, max 3 rounds. Targeted tool regeneration via LLM (reuses `_ai_generate_tool_impls` patterns). Repair tokens tracked per round.
- **`--no-postgen-review` CLI flag** — Skips Phase 3.5 self-review for users who want raw generation only. Default: postgen review runs after Phase 3.
- **PostGenReviewSummary in HatchReport** — New `postgen_review` field on `HatchReport` (Pydantic). Renders as a "Post-Generation Review" panel in terminal output and as a JSON object in `--json` output. Verdict (READY/WARN) propagates to the top-level `compute_verdict()` — WARN if postgen verdict is WARN. Token usage from repair LLM calls accumulates into `total_tokens`.
- **48-test postgen_review test suite** — `tests/test_postgen_review.py`. Covers each inspection check type, side-effect detection branches, iteration termination conditions, HatchReport integration, and detection capability verified with synthetic tools.py fixtures containing each known bug pattern (currency-converter NameError, minimal-skill AttributeError, data-analyzer logic-error limitation).
- **Agent-level context in repair LLM** — `_regenerate_tool_via_llm()` now emits an `=== AGENT CONTEXT ===` block (identity.display_name, identity.purpose, intent.summary, intent.triggers, base.archetype) alongside the existing TOOL DEFINITION / SKILL CONTEXT / DETECTED BUGS sections. Previously the repair LLM only saw single-tool metadata, missing agent-wide semantics (e.g. archetype constraints, triggers). System prompt adds a rule: "Honor the AGENT CONTEXT: the repair must fit the agent's intent, triggers, and archetype". Closed-loop verified on minimal-skill: round 1 detects `text.strip()` AttributeError, round 2 LLM repair (with agent context) returns `if text is None: return ""` and verdict flips WARN → READY.
- **5-test repair-application regression suite** — `TestRebuildFunctionSource` (3 tests) and `TestApplyToolRepair` (2 tests) covering multi-line docstring preservation, single-line docstring, no-docstring, missing-function, and end-to-end repair application. Locks in the `end_lineno + 1` fix below.

### Fixed

- **`_rebuild_function_source` dropped multi-line docstring closer** — `doc_end_idx` was computed as `doc_node.end_lineno - node.lineno` (exclusive), but `end_lineno` is inclusive — the closing `"""` line was sliced off, producing unparseable code (`SyntaxError: unterminated triple-quoted string`). Repair LLM calls succeeded (returned valid body), but `_apply_tool_repair` silently failed at the `ast.parse(new_content)` check, so tools.py was never updated and the iteration loop terminated at round 1 with verdict WARN. Fixed by `+1` to make the bound exclusive. This was the root cause of the data-analyzer 4-bug "all repairs failed" symptom in the prior closed-loop test.
- **`_build_agent_context` schema mismatch (dead code)** — `identity.purpose` and `base.archetype` are NOT fields in the `Identity` / `BaseSpec` Pydantic models (`spec.py`). The original implementation read `ahs_dict["identity"]["purpose"]` and `ahs_dict["base"]["archetype"]` — both always returned empty, making the AGENT CONTEXT block's `Purpose:` and `Archetype:` lines dead code and the system_prompt's "MULTI_STEP agents should keep state across calls" rule a no-op. Fixed by: (1) removing `purpose` (never existed), (2) passing `archetype` as an explicit parameter from `hatch_command` (which owns the `classification` object from `classify_skill()`). New `iterate_until_gate(..., archetype=str | None)` and `_run_postgen_review(..., archetype=str | None)` signatures. `hatch_command` extracts `classification.archetype.value` and passes it through.
- **`_replace_function_body` did not handle `ast.AsyncFunctionDef`** — async tool functions (`async def fetch_data(...)`) never matched the `isinstance(node, ast.FunctionDef)` check, so async tool repair silently failed (returned `False`, no body replacement). Fixed by accepting both `ast.FunctionDef` and `ast.AsyncFunctionDef`. The `async def` prefix is preserved in the rebuilt source because `ast.get_source_segment` returns the full original function text.
- **`_replace_function_body` log noise on multi-file iteration** — When `_apply_tool_repair` iterates multiple candidate `tools.py` files, "function not found in this particular tools.py" is normal control flow (the function lives in one file, not all). Previously emitted `logger.warning` for every non-matching file. Downgraded to `logger.debug` to keep warning logs meaningful.
- **`test_repair_via_llm_fixes_undefined_var` weak assertion** — Test asserted `verdict in (READY, WARN)`, which passed even when repair failed (regression). Tightened to `verdict == READY` with diagnostic message showing findings on failure, since the mock returns valid Python that should fix the bug.

### Changed

- **sentence-transformers is now an optional dependency** — `pip install agenthatch` no longer pulls PyTorch. Core install includes BM25 keyword search only. For semantic (embedding) search: `pip install agenthatch[semantic]`. `_ensure_embedder()` handles ImportError gracefully, falling back to keyword-only mode.
- **HatchReport.compute_verdict() now considers postgen_review** — Adds a fourth WARN trigger: `postgen_review.verdict == "WARN"`. The verdict remains advisory (PASS/WARN only — no FAIL, never blocks).

### Fixed

- **CI mypy: sentence_transformers import-not-found** — After moving sentence-transformers to optional dependency, CI's mypy --strict could not find the module. Added `[[tool.mypy.overrides]] module = "sentence_transformers" ignore_missing_imports = true` to pyproject.toml. Works in both CI (module absent) and local (module present) without unused-ignore errors.
- **CI mypy: click.shell_completion import-not-found** — Pre-existing mypy error surfaced after CI environment change. Added mypy override for `click.shell_completion`. Replaced `# type: ignore[no-any-return]` on `completer.source()` with explicit `str()` conversion for environment-agnostic type safety.
- **Documented edge case: empty plan is_complete** — `StructuredPlan.is_complete` returns True for empty plan due to vacuous truth (`all([]) == True` in Python). Test documents this as expected behavior.
- **Documented limitation: _update_topology retroactive update** — `_update_topology` only records requires at `add_entry` time if the provider already exists. Does not retroactively update existing entries when a new provider is added. Circular dependency test uses manual topology construction to test Kahn's algorithm directly.
- **`HatchReport.to_terminal` temperature-range caption hardcoded lower bound** — The Harness Detail table caption read `(provider range 0–{hi:g})` with a hardcoded `0` instead of the actual `lo` from `temperature_range`. The `lo` variable was unpacked but never used (dead code). Fixed by formatting `{lo:g}–{hi:g}`. Latent because all current providers (OpenAI/DeepSeek/Anthropic) have `lo=0.0`; would surface if a provider with a non-zero lower bound is added.
- **`_probe_mcp_server` docstring/implementation mismatch** — Docstring claimed "Returns False if server unreachable OR if any tool has empty schema", but the implementation logged empty-schema tools as warnings and still returned `True`. Two fix directions considered: (A) make implementation match docstring by returning `False` on empty schema — rejected because it would conflate schema quality with network reachability, surfacing misleading "check VPN" warnings for what is actually a schema issue; (B) fix docstring to match implementation — adopted. `mcp_reachable` now documents that it reflects network reachability only; schema quality is a separate advisory dimension logged via `logger.warning`. Both dimensions remain advisory per v0.8.10 "never block" philosophy.
- **`_json_type_to_python` mapped JSON Schema `number` to `int`** — Per JSON Schema spec, `number` is any numeric value (including floats) and `integer` is a subset. The mapping `"number": "int"` caused generated tool signatures to annotate float parameters (e.g. `threshold: number` for IQR outlier detection, commonly `1.5`) as `int`, losing float precision and misleading IDE/mypy. Fixed to `"number": "float"`. Verified against `tests/fixtures/skills/data-analyzer/agenthatch.yaml` which uses `type: number` for 11 statistical fields.

---

## [v0.9.21] — 2026-06-29

### Added

- **hatch report: harness temperatures** — report now displays per-harness temperature values alongside confidence and reasoning traces
- **zero critic-role temperature** — reflection/critic harness temperature set to 0.0 for deterministic validation output

### Fixed

- **fix: match harness .name in should_skip_reflection** — `should_skip_reflection()` compared harness keys against `.name` attribute but harness identifiers are stored as dict keys ("A", "B", etc.), not `.name`. Caused reflection to run on harnesses that should have been skipped (e.g., Harness A/F with confidence ≥ 0.9), wasting tokens. Now matches against dict keys consistently.

---

## [v0.9.20] — 2026-06-25

### Added

- **reflection loop wired into orchestrator** — v0.9.20 connects the `reflect_and_correct_harness()` function (previously standalone) into the engine orchestrator at two points: Step 5.5 (A/B/C/D/F harnesses reflect against SKILL.md and peer outputs) and Step 6.5 (Harness E reflects on the assembled AHSSPEC). Completes the "self-review" half of ROADMAP Phase 1.
- **fidelity checkpoint** — `run_fidelity_checkpoint()` added at post-assembly stage, scoring AHSSPEC fidelity against source SKILL.md
- **hardened apply_corrections** — `apply_corrections()` now uses dot-path field targeting (e.g., `"intent.triggers"`) for precise correction application, avoiding full-spec regeneration

---

## [v0.6.0] — 2026-06-05

### Architecture Transformation: "Agent Factory"

v0.6 marks a major architectural transformation from "configuration-driven" to "Agent Factory" mode. The core runtime has been extracted into a standalone `agenthatch-core` package, and the `hatch` command now includes built-in Phase 3 agent generation.

### Added

- **agenthatch-core**: New standalone package providing the universal agent runtime base
  - `AHCoreAgent`: Base class for all generated agents
  - `LLMClient`: Unified LLM call interface (OpenAI, DeepSeek, custom providers)
  - `CapBus`: Capability bus for tool registration, routing, and execution
  - `Sandbox`: Subprocess sandbox with command whitelisting and timeout control
  - `ConversationLoop`: LLM ↔ Tool conversation loop with circuit breaker and retry
  - `ContextManager`: System prompt builder, history management, auto-compaction
  - `CompactSummary`: LLM-generated structured context compression
  - `resolve_runtime_config()`: Environment variable resolver with `${VAR}` syntax
- **Phase 3 Agent Generation**: `hatch` command now generates standalone, independently-runnable Agent directories
  - Jinja2 template engine with 6 templates (pyproject.toml, agent.py, cli.py, tools.py, runtime.toml, README.md)
  - `GenerateEngine` class for extracting AHSSPEC variables and rendering templates
  - `generate_agent()` convenience function
- **`agenthatch run` redesign**: Direct agent launching via in-process import with Rich Live TUI
  - Three-level agent discovery: current dir → skillhouse index → user dir
  - Interactive commands: `/help`, `/compact`, `/clear`, `/quit`
- **`agenthatch migrate`**: New command for migrating v0.5 agenthatch.yaml to v0.6 format
- **`agent.status` and `agent.generated_at` fields**: New metadata fields in agenthatch.yaml
- **`agent_output` field**: New field in skillhouse index for tracking agent generation paths

### Changed

- **agenthatch.yaml format**: Runtime configuration (`agent.runtime.*`) removed and migrated to `runtime.toml`
- **`hatch` command**: Now executes full 3-phase pipeline by default (parse → harness → generate)
  - `--no-generate` flag to skip Phase 3 (review mode)
  - `--force` flag to overwrite existing output
  - `--dry-run` flag to preview without writing
  - `--no-copy-skills` flag to exclude original SKILL.md
- **Dependency architecture**: `agenthatch` now depends on `agenthatch-core>=0.6.0` (one-way dependency)
- **`ConversationLoop`**: Migrated to `agenthatch-core`, now receives `llm`, `capbus`, `sandbox`, `ctx` as constructor parameters
- **`ContextManager`**: Migrated to `agenthatch-core`, accepts `dict` or `SpecProtocol` for spec
- **`LLMClient`**: Migrated to `agenthatch-core`, accepts provider details directly

### Fixed

- **Fix-1**: hatch exit code verification — exit code is now always 0 on success
- **Fix-2**: init command version number — now reads from `agenthatch.__version__` instead of hardcoded string
- **Fix-3**: skills list display — unhatched skills now show `[dim]not hatched[/dim]` instead of `Version ?`
- **Fix-4**: reasoning_content handling — verified DeepSeek V4 Pro streaming with reasoning content fallback
- **Fix-5**: TUI backspace key — Rich Live context properly paused during `Prompt.ask()` input
- **Fix-6**: legacy `run` command logic — removed configuration-driven path, replaced with agent direct-launch

### Removed

- **`agent.runtime` fields** from agenthatch.yaml (provider, model, api_key, temperature, max_tokens, features, compact)
- **Legacy `SkillAgent.from_ahspec()` runtime assembly path**: Replaced by `AHCoreAgent` + generated agent code
- **Configuration-driven `run` path**: Replaced by agent direct-launch mode

### Deprecated

- `agent.runtime` in agenthatch.yaml: Issues `DeprecationWarning` on load, still functional
- Will be removed in v1.0.0 per the deprecation schedule

---

## [v0.9.19] — 2026-06-22

### Fixed (accumulated bug fixes)

- **fix: pass through temperature/max_tokens in chat_structured fallback** — `chat_structured()` Instructor fallback path hardcoded `temperature=0.0` and `max_tokens=4096`, discarding caller-configured values. Harnesses configure per-task values (e.g. AssembleHarness uses 8192) that were silently overridden. Now passes through the function parameters.
- **fix: remove dead harness timeout code** — `_build_harnesses()` computed `d_timeout` based on client features but never used it (only logged then discarded). Removed the dead code.
- **fix: resolve numpy 2.5.0 mypy incompatibility in CI** — numpy 2.5.0 stubs use `type` statement (Python 3.12+) which breaks mypy when `pyproject.toml` hardcodes `python_version = "3.11"`. CI now passes `--python-version` from matrix to mypy; `pyproject.toml` adds mypy override to ignore numpy module errors.
- **fix: remove global sys.stderr hijack in _ensure_embedder** — `_ensure_embedder` replaced process-level `sys.stderr` with `StringIO` during SentenceTransformer download (up to 60s). This silently discarded all other threads' stderr output. On timeout, the hijack persisted until the daemon thread finished. Removed the hijack; `hf_log.setLevel(ERROR)` already suppresses Python logging noise, and C-level stderr (SSL errors, etc.) is preserved as diagnostic signals.
- **fix: checkpoint migration dead code** — `CheckpointManager.__init__` mkdirs `new_dir` before the migration check, so `not new_dir.exists()` was always `False` — the `shutil.copytree` migration never ran. Users upgrading from old checkpoint paths (`~/.agenthatch/sessions/`) silently lost history. Migration now runs before `CheckpointManager()` construction, and the condition checks `checkpoint.json` existence instead of directory existence.

---

## [v0.9.18] — 2026-06-21

### Fixed (accumulated bug fixes)

- **fix: add missing ThinkingDelta import in LLMClient.chat_stream** — `ThinkingDelta` was referenced but not imported, causing `NameError` when streaming reasoning content from DeepSeek V4 Pro. Fixed via deferred import to avoid circular dependency.
- **fix: correct split count in MCPClient.register_with_capbus** — `split("__", 1)` should be `split("__", 2)` for three-segment MCP tool names (`mcp__<server>__<tool>`). The wrong split produced incorrect server names in the server-side tool discovery path.
- **fix: correct escape sequence in _escape_fts5_query** — `re.sub` replacement had 6 backslashes (3 literal `\`, capture group lost) instead of 3 (1 literal `\` + capture group). FTS5 special characters were not properly escaped, silently falling back to LIKE search.
- **fix: use getattr for reasoning_tokens in DirectLoop._record_usage** — `DirectLoop` accessed `usage.reasoning_tokens` directly, but OpenAI's `CompletionUsage` nests it under `completion_tokens_details`. This caused `AttributeError` for prompt-only skills. Now uses `getattr(usage, "reasoning_tokens", 0)` matching `ConversationLoop`.

---

## [v0.9.16] — 2026-06-17

### Open Source Prep: Final Polish

- **Remove Discord links** from README, README_CN, SUPPORT.md, CONTRIBUTING.md — defer to D+7~14 when community exists (empty room problem)
- **Add GitHub Release auto-creation** to publish.yml via `softprops/action-gh-release@v2` — tag push now creates both PyPI artifact and GitHub Release
- **Add .gitignore entries** for `.workbuddy/`, `campaign/`, `deliverables/` — marketing artifacts excluded from package

---

## [v0.9.14] — 2026-06-17

### Community & CI Fixes

- **Add Discord and Twitter/X links** to README, SUPPORT.md, CONTRIBUTING.md
- **Fix CI**: add `types-PyYAML>=6.0` to `[dev]` dependencies — GitHub Actions failed on mypy with missing yaml stubs
- **Humanizer polish**: remove AI writing patterns from README (em dash overuse, inflated language)

---

## [v0.9.13] — 2026-06-17

### README Audit & CI Infrastructure

#### Added
- **CI workflow** (`.github/workflows/ci.yml`): ruff lint + mypy --strict + pytest on Python 3.11/12/13 matrix
- **Publish workflow** (`.github/workflows/publish.yml`): PyPI trusted publishing via OIDC, triggered on `v*` tag push
- **README_CN.md**: Chinese translation of README

#### Fixed (README — source code audit)
- Remove architecture diagram placeholder (CLI tools don't need diagrams; text pipeline + harness table is sufficient)
- Remove docs site TODO (docs site deferred, README is the documentation for CLI tools)
- Fix "generates cli.py" claim → actual output is `agent.py` + `tools.py` + `references.py`
- Fix "run concurrently" → harnesses run sequentially (A→B→C→D→F→E)
- Add `references.py` to output file tree (was missing)
- Fix file paths: outputs live under `src/{package_name}/`, not root
- Fix determinism claim: "Same SKILL.md → same agent binary" → "Same SKILL.md → same AHSSPEC structure (low-temp inference)"

#### Removed
- Demo section from README (Quick Start is the demo for CLI tools)
- Star History chart (pre-launch anti-pattern)
- "Coming soon" shields badges for non-existent Discord/Twitter

---

## [v0.5.10] — 2026-05-XX

### Fixed
- Empty response handling in OpenAI-compatible providers
- Checkpoint TypeError in context compaction
- Token budget inflation during conversation loop
- MCP server URL extraction from skill body
- Reasoning content extraction from streaming responses
- Structured chat reasoning fallback
- Harness E assembly confidence scoring
- Token adjustment log level verbosity
- Multi-format content extraction in LLM responses

---

## [v0.5.0] — 2025-XX-XX

### Added
- SKILL.md parsing (Phase 1): Deterministic frontmatter + content parsing
- LLM Harness reasoning (Phase 2): 6-agent harness pipeline for AHSSPEC generation
- agenthatch.yaml output: Structured skill specification
- `agenthatch hatch` command: SKILL.md → agenthatch.yaml pipeline
- `agenthatch run` command: Interactive agent conversation
- `agenthatch init` command: First-time setup wizard
- `agenthatch skills` command: Skill listing and management
- Skillhouse index: Skill discovery and registration
- Semantic search: sentence-transformers based skill retrieval
- Rich TUI: Live streaming with tool call visualization

---

## [v0.2.0] — 2025-XX-XX

### Added
- Initial project scaffolding
- Basic CLI framework with Typer
- Configuration management