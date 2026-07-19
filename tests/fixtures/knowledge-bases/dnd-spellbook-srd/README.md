# D&D 5e Spellbook Sage — Test Knowledge Base

This knowledge base is a hand-curated reference of Dungeons & Dragons 5e
spells, used to stress-test the agenthatch v1.0.1 KB pipeline against
English markdown content with realistic structure (frontmatter-style
metadata blocks, multi-section descriptions, cross-spell references).

## Licensing Note

The spell descriptions below are original paraphrases inspired by the
5e SRD (released under OGL 1.0a).  They are written from scratch to
avoid copying any specific WotC wording while keeping the rules
mechanics intact.  Treat this KB as a test fixture, not a replacement
for the published SRD.

## Layout

- `spells/` — 12 spell entries, ranging from short cantrips (~250 chars)
  to long multi-paragraph spells (~800+ chars) to exercise chunking
  boundaries.
- `draft/` — holds a single `secret-unreleased-spell.md`.  The skill's
  `exclude_patterns` should drop it from both the index AND the
  B3/B4 LLM context (Bug #3 regression check).
