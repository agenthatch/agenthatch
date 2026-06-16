# Roadmap

This is the planned direction for agenthatch. Items within a phase are
unordered — the order may shift based on feedback and contribution.

---

## Phase 1: Quality & Observability

### AI self-review loop

Harnesses already run Analyze → Infer → Self-Validate → Correct internally.
Extend this to a post-generation review phase: the hatched agent inspects its
own code, tests its own tools, and iterates until the quality gate passes
autonomously.

### Hatch report (`agenthatch report`)

Every hatch run produces a structured report — harness confidence scores,
reasoning traces, degradation events, token consumption per phase, and a
pass/fail verdict. Readable as both terminal output and JSON for CI pipelines.

---

## Phase 2: Intelligence

### Knowledge-backed agents (RAG-native skillagent)

Agents that ship with their own vector index. A medical SKILL.md doesn't just
describe a diagnosis workflow — the hatched agent embeds a full knowledge base,
retrieves relevant references per query, and can be shared as a self-contained
package ready for production.

---

## Phase 3: Composition

### Skill fusion (`agenthatch hatch --fuse`)

Feed the pipeline multiple SKILL.md files. The harness detects overlapping
domains, resolves conflicts, and produces a single fused agent that combines
capabilities from all inputs. One agent that understands both git workflows
and deployment pipelines, without leaking instructions between them.

### Meta-agent (`agenthatch all`)

A top-level agent that knows about every agent in your skillhouse. You talk
to one interface — it routes tasks to the right hatched agent, collects results,
and synthesizes a response. Think of it as Claude Code, but backed by an army
of specialized agents instead of one monolithic system prompt.

---

## Phase 4: Ecosystem

### Agent Marketplace

A registry where hatched agents can be published, discovered, and installed.
Versioned, reproducible, shareable — `agenthatch install user/medical-agent`
should work like `pip install`.

### Multi-channel communication

Hatched agents that connect to WhatsApp, Telegram, Slack, Discord, and other
messaging platforms — similar to OpenClaw's channel model. An agent doesn't
just run in a terminal; it lives where your team already communicates.

### Docker sandbox mode

The current sandbox runs subprocesses directly with a command whitelist.
Add an optional Docker-backed execution layer for full filesystem and network
isolation — safe enough to run untrusted tool code in production.

---

## Final milestone: One-sentence agent

```
agenthatch hatch "monitor this repo and open an issue when CI fails"
```

No SKILL.md required. A single sentence → full pipeline → runnable agent. The
harness infers identity, intent, interface, and base from a natural language
description. This is the north star — everything in Phases 1–4 builds toward it.

---

## What's already here

Some items people commonly ask about are already implemented:

| Feature | Status |
|---|---|
| **PlanLayer state machine** (STARTING → DONE) | ✅ In `agenthatch-core` since v0.9.8 |
| **Subprocess sandbox** with command whitelist | ✅ In `agenthatch-core` (STANDARD + EXTENDED tiers) |
| **6-Harness LLM pipeline** with self-validation | ✅ Core pipeline since v0.6 |
| **MCP auto-detection** (Harness F) | ✅ Since v0.7 |
| **Context auto-compaction** | ✅ In `agenthatch-core` context manager |

---

## Contributing to the roadmap

This is a living document. If something here matters to you — or if something
missing should be here — open a [GitHub Discussion](https://github.com/agenthatch/agenthatch/discussions).
