# Roadmap

This is the planned direction for agenthatch. Items within a phase are
unordered — the order may shift based on feedback and contribution.

---

## Phase 1: Quality & Observability — ✅ Concluded in v0.9.23

The post-generation self-review loop shipped in v0.9.22. After Phase 3
generation, the hatched agent inspects its own `tools.py` for undefined
variables, None-attribute bugs, and semantic stubs. Each tool runs once
with mock parameters to catch runtime errors. When bugs are found, the LLM
regenerates the broken function body and re-inspects (up to 3 rounds).

Phase 1 is concluded. Proposed observability extensions (per-round token
counts, iteration traces, repair diffs) are deferred. They're nice to have
but not on the critical path. Open a Discussion if a specific signal is
needed.

---

## Phase 2: Intelligence — ✅ Shipped in v1.0.0

### Knowledge-backed agents

Agents that ship with their own knowledge base. A medical SKILL.md doesn't
just describe a diagnosis workflow. The hatched agent embeds a full knowledge
base at hatch time, retrieves relevant references per query at runtime, and
can be shared as a self-contained package ready for production.

v1.0.0 ships the first cut:

- SQLite FTS5 + BM25 keyword retrieval, with thread-local connections to
  keep cross-thread access safe.
- `knowledge_base.py.j2` template generates a runtime `retrieve()` function
  whose `WHEN_TO_RETRIEVE`, `QUERY_TEMPLATES`, and `SYSTEM_PROMPT_SECTION`
  are inferred by the LLM from the skill at hatch time.
- `KnowledgeBaseBrick` assembles the retrieve tool into CapBus at agent
  startup; the system prompt gets a KB section.
- Optional semantic (embedding) search behind `pip install agenthatch[semantic]`.
  Core install keeps BM25 only, so `pip install agenthatch` no longer pulls
  PyTorch.

v1.0.1 stabilizes the KB agent runtime. A bare `yield from` in generated
`chat_stream` was discarding the subgenerator's return value, so callers
got an empty string even though the answer had streamed to the terminal.
`kb_max_text` had a typo in the streaming path that silently disabled
meta-narration stripping. Both fixed, plus a wider pattern list for the
trailing meta-narration the LLM tends to append before `task_complete`.

v1.0.2 through v1.0.4 are the KB hardening pass. The chunker was silently
dropping text when a paragraph exceeded `chunk_size`; `discover_kb_files`
was ignoring `exclude_patterns`; the generated `pyproject.toml` was
missing `agenthatch-core` so `retrieve()` failed silently after pip
install; `retrieve(top_k=0)` returned one result instead of zero; and
`_fuse_results` leaked zero-score documents into top-k at `alpha=1.0`
or `alpha=0.0`. Regression coverage for the KB module went from 29 to
48 tests, plus 12 static guard tests that prevent template regressions
(`return (yield from`, `kb_max_text`, `sys.modules` registration) and
5 MCP/LLM regression tests.

v1.0.5 shifts the harness pipeline. Harness E now merges MCP server
definitions from F's output into the assembled interface, so generated
agents ship with MCP tools wired in without a separate step. Confidence
scores take a 5% hit per retry (floored at 0.75) so a harness that
succeeds on retry 3 doesn't report the same confidence as one that
nailed it on the first try. JSON Schema `enum`/`min`/`max`/`pattern`
constraints are now mapped to Pydantic types instead of silently
dropped.

v1.0.6 fixes two crashes. `KnowledgeStore()` was instantiated before
the `try` block, so an init failure (bad model path, disk full) made
the `finally` block crash on `store.close()` with an unbound variable.
Generated KB agents were missing `from .knowledge_base import retrieve`,
so the first retrieve call raised `NameError`. The E harness failure
path also stopped freezing the spinner.

v1.0.7 adds exception-swallow detection to the post-generation review.
Generated tools that catch `Exception` and return `None` / `""` / `[]`
/ `pass` make real failures look identical to legitimate empty results.
WARNING-only on purpose — an earlier attempt let B4 rewrite these
automatically, but the LLM kept either deleting legitimate fallbacks or
narrowing to a specific exception class that missed the actual failure
mode.

What's not done yet: LLM re-rank of retrieved chunks, cross-agent shared
KB memory, and a maintenance loop that flags stale entries. The hybrid
search is BM25 + embeddings; re-ranking is still on the to-do list.

---

## Phase 3: Composition

### Skill fusion (`agenthatch hatch --fuse`)

Feed the pipeline multiple SKILL.md files. The harness detects overlapping
domains, resolves conflicts, and produces a single fused agent that combines
capabilities from all inputs. One agent that understands both git workflows
and deployment pipelines, without leaking instructions between them.

### Meta-agent (`agenthatch all`)

A top-level agent that knows about every agent in your skillhouse. You talk
to one interface; it routes tasks to the right hatched agent, collects
results, and synthesizes a response. Think of it as Claude Code, but backed
by an army of specialized agents instead of one monolithic system prompt.

---

## Phase 4: Ecosystem

### Agent Marketplace

A registry where hatched agents can be published, discovered, and installed.
Versioned, reproducible, shareable — `agenthatch install user/medical-agent`
should work like `pip install`.

### Multi-channel communication

Hatched agents that connect to WhatsApp, Telegram, Slack, Discord, and other
messaging platforms, similar to OpenClaw's channel model. An agent doesn't
just run in a terminal; it lives where your team already communicates.

### Docker sandbox mode

The current sandbox runs subprocesses directly with a command whitelist.
Add an optional Docker-backed execution layer for full filesystem and
network isolation, safe enough to run untrusted tool code in production.

---

## Final milestone: One-sentence agent

```
agenthatch hatch "monitor this repo and open an issue when CI fails"
```

No SKILL.md required. A single sentence goes through the full pipeline and
produces a runnable agent. The harness infers identity, intent, interface,
and base from a natural language description. This is the north star;
everything in Phases 1 through 4 builds toward it.

---

## What's already here

Some items people commonly ask about are already implemented:

| Feature | Status |
|---|---|
| PlanLayer state machine (STARTING → DONE) | ✅ In `agenthatch-core` since v0.9.8 |
| Subprocess sandbox with command whitelist | ✅ In `agenthatch-core` (STANDARD + EXTENDED tiers) |
| 6-Harness LLM pipeline with self-validation | ✅ Core pipeline since v0.6 |
| MCP auto-detection (Harness F) | ✅ Since v0.7 |
| Context auto-compaction | ✅ In `agenthatch-core` context manager |
| Hatch report (`--report` / `--json`) | ✅ Since v0.9.17 |
| Post-generation self-review (inspect → test → repair loop) | ✅ Since v0.9.22 |
| KnowledgeBaseBrick (RAG retrieval, SQLite FTS5 + BM25) | ✅ Since v1.0.0 |
| KB pipeline hardening (chunker, exclude, top_k, fuse, path resolution) | ✅ Since v1.0.4 |
| Confidence retry penalty + JSON Schema constraint mapping | ✅ Since v1.0.5 |
| Exception-swallow antipattern detection in postgen review | ✅ Since v1.0.7 |

---

## Contributing to the roadmap

This is a living document. If something here matters to you, or if something
missing should be here, open a [GitHub Discussion](https://github.com/agenthatch/agenthatch/discussions).
