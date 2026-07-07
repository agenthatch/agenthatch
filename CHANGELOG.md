# CHANGELOG

All notable changes to agenthatch will be documented in this file.

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