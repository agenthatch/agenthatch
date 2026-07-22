# CHANGELOG

All notable changes to agenthatch will be documented in this file.

---

## [Unreleased]

### Fixed

- **Bug #9: `_fuse_results` 在 `alpha` 极端值下泄漏零分结果** — `KnowledgeStore._fuse_results(bm25_results, emb_results, alpha)` 的加权公式为 `final_score = alpha * keyword + (1-alpha) * embedding`。当用户显式设 `alpha=1.0`（纯关键词模式）时，原实现仍遍历 embedding 结果并以 `(1-1.0)*score = 0` 加入 fused dict，导致 emb-only 文档以零分占据 top-k 槽位；`alpha=0.0`（纯 embedding 模式）对称地泄漏 bm25-only 零分结果。修复为在 `alpha >= 1.0` 时直接 early-return 仅含 BM25 归一化结果，`alpha <= 0.0` 时对称 early-return 仅含 embedding 结果；若"纯"侧为空，仍回退到另一侧（比返回空更好）。混合 alpha 区间（0 < alpha < 1）保留原有 v1.0.0 融合逻辑不变。新增 8 个回归测试在 `tests/test_kb_regressions.py::TestBug9FuseResultsNoLeakAtAlphaExtremes`。

---

## [v1.0.1] — 2026-07-16

### Fixed

- **R4-V23: `chat_stream()` 裸 `yield from` 丢弃子 generator 返回值** — 生成的 agent 在 `chat_stream` 中使用 `yield from super().chat_stream(user_input)`，Python 会把子 generator 的 `return` value 丢弃（`yield from` 表达式的值为 `None`，除非显式写 `return (yield from ...)`）。导致 `ConversationLoop.stream()` 老老实实 return 的 `final_text`（如 466 字符的完整答案）在调用方拿到的却是空字符串。文本已流式输出到终端，但编程接口返回空。修复为 `return (yield from super().chat_stream(user_input))`，同步修复 `agent.py.j2` 模板防止复发。
- **R4-V22: `chat_stream()` 路径 `kb_max_text` typo** — `agent.py` 中 `chat_stream()` 路径的 `kb_max_text = 1`，注释写着 "Same KB auto-continuation cap as chat()"，但 `chat()` 路径实际是 `0`。这一个数字让流式路径下 `self._max_consecutive_text_only == 0` 永远不成立，`_strip_trailing_meta_narration` 从未被调用，meta-narration（"前已详答,此不赘述"、"前问已答毕"）泄漏到用户。修复为 `0`，与 `chat()` 路径一致。
- **R4-V22: `_strip_trailing_meta_narration` 旧策略误删正文** — 旧策略"在最早匹配点截断到末尾"会删除 meta-narration 之后的全部正文（如结尾的"阁下若欲探询..."邀请句），触发 40% 安全保护，反而让 meta-narration 泄漏通过。重写为"删除包含匹配的整个句子"策略：找到所有匹配，扩展到完整句子边界（。！？\n），合并重叠区间，只删这些句子。安全阈值从 40% 调整为 50%。
- **R4-V22: meta-narration 模式列表扩展** — 添加"前问已答"、"已答毕"、"前文已..."、"上文已..."、"已详答"、"已作答"等 LLM 变体；`前已` 模式中 `详细` 改为 `详细?` 使"前已答"也能匹配。
- **R4-V21: meta-narration 残留在 text stream** — KB agent 在调用 `task_complete` 之前，LLM 常加几句"已完整解答...无剩余步骤"的 meta-commentary，尽管 B4 (e) 明确禁止。引入 `_strip_trailing_meta_narration()` 函数，在最后 600 字符中匹配 meta-narration 模式并删除整句。
- **R4-V20: `task_complete` 重复 yield meta-summary** — `task_complete` 被调用时，若已流式输出过真实答案（`has_yielded_text=True`），仍会再次 yield `summary` 参数内容（通常是"已回答..."的 meta-summary），导致用户看到答案后又看到一句冗余总结。修复为：`has_yielded_text=True` 时用 `accumulated_text` 作为 `final_text`，不再 yield summary。
- **R4-V16: KB 包名解析走 MRO 导致 import 失败** — `type(self).__module__` 在某些加载路径下走 MRO 匹配到基类 `agenthatch_core.agent`（其 `__package__` 是 "agenthatch_core"），导致 `importlib.import_module(f"{pkg}.knowledge_base")` 报 `No module named 'agenthatch_core.knowledge_base'`。修复为：直接从 `__module__` 派生包名，并在 `run.py` 中于 `exec_module` 前将 agent module 注册到 `sys.modules[spec.name]`。

### Changed

- **KB agent auto-continuation 抑制** — 引入 `max_consecutive_text_only` 和 `nudge_grace` 参数（R4-V17），KB agent 传 `0` 让循环在首次 text-only 响应后即返回，避免 auto-continuation 产生重复答案和 meta-summary。

---

## [v1.0.0] — 2026-07-14

### Added

- **KnowledgeBaseBrick（RAG 检索）** — agent 工程意义上的工程知识库，区别于 skill 内部的 `references/` 共生知识。用户通过 CLI 第二参数指定 KB 路径，孵化期构建向量索引，运行时通过 `retrieve` 工具检索。
  - **Phase B（编译期集成）**：KB inference pipeline（B2 检测 → B3 用法策略 → B4 prompt 生成）。`_build_knowledge_index()` 在 `_prepare_output_dir` 之后、模板渲染之前运行。
  - **Phase C（运行期装配）**：`RetrieveTool` 注册到 CapBus；`AHCoreAgent` KB assembly block；`ContextManager` 注入 KB system prompt。
  - **`knowledge_base.py.j2` 模板**：生成 runtime `retrieve()` 函数，含 LLM-inferred 的 `WHEN_TO_RETRIEVE`、`QUERY_TEMPLATES`、`SYSTEM_PROMPT_SECTION`。
  - **SQLite FTS5 索引 + BM25 评分**：FTS5 索引将 `-` 替换为空格避免 NOT 操作符解析；BM25 评分用 `abs(rank)/(1+abs(rank))` 优先相关文档。
- **HatchReport confidence 来源统一** — Hatch Summary 的 confidence 值改用 E harness cross-evaluation 分数，self-assessment 作为 fallback，修复了 Confidence panel（1.00）与 Hatch Summary（0.50）不一致的问题。
- **Phase 3/3 标题** — 正常（非 dry-run）hatch 流程在控制台输出 `▸ Phase 3/3 Agent Generation` 标题。

### Fixed

- **`--force` 误删 KB 索引** — `--force` flag 在覆盖输出目录时误删 KB 索引文件。修复为 `--force` 不再清除 KB 索引。
- **SQLite 跨线程错误** — SQLite 连接改用 thread-local storage，避免跨线程访问错误。
- **BM25 评分反转** — BM25 评分未取绝对值，负 rank 导致评分异常。修复为 `abs(rank)/(1+abs(rank))`。
- **FTS5 hyphen 解析** — FTS5 将 hyphen 解析为 NOT 操作符导致查询失败。修复为索引时将 `-` 替换为空格。
- **B2 detector 误报** — B2 detector 通过目录名匹配误判非 KB skill 为 KB skill。修复为只识别 KB 特定词汇，不匹配目录名。

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