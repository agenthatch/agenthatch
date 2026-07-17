"""Knowledge base inference pipeline (v1.0.0 Phase 2.5).

Three-stage pipeline that runs between Phase 1 (context assembly) and
Phase 2 (AgentHarness inference) when the user passes a knowledge base
path via ``agenthatch <skill> <knowledgebase>``.

Stages:
  B2 ``detect_kb_mention()`` — Deterministic regex + filename scan of
     SKILL.md to check whether the skill explicitly references the KB.
     Returns ``KBDetectorResult`` with evidence.
  B3 ``infer_kb_usage()`` — LLM-driven inference (temp=0.3) of *when*
     and *how* the agent should retrieve from the KB.  Only runs when
     B2 reports no explicit mention — this is the "fallback" path for
     skills that don't document their KB usage.
  B4 ``generate_kb_prompt()`` — LLM-driven generation (temp=0.2) of
     the system-prompt section + ``retrieve`` tool description that
     will be injected into the runtime agent.

The orchestrator ``run_kb_pipeline()`` chains B2 → B3 → B4 and returns
a fully populated ``KnowledgeBaseConfig`` ready to attach to AHSSPEC.

Design references:
- ``docs/v1.0.0-knowledge-base-iteration-plan.md``
- Existing ``postgen_review.py`` for the B2/B3/B4 naming convention
- OpenClaw's KB skill for the small-chunk + auto-tag pattern
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agenthatch.skill.spec import (
    KBPromptArtifact,
    KBUsageStrategy,
    KnowledgeBaseConfig,
    KnowledgeBaseSource,
)

logger = logging.getLogger("agenthatch")

# ─────────────────────────────────────────────────────────────────────────
# Default file patterns for KB directory sources
# ─────────────────────────────────────────────────────────────────────────

_DEFAULT_INCLUDE_PATTERNS: tuple[str, ...] = ("*.md", "*.txt", "*.rst", "*.markdown")
_MAX_KB_FILES_PREVIEW = 20      # how many file names to show the LLM
_MAX_KB_SAMPLE_CHARS = 4000     # how much SKILL.md body to send to the LLM

# v1.0.1 (R3-M14): Hard cap on LLM call duration.  B3/B4 inference
# calls an injected ``chat_fn`` (typically ``LLMClient.chat``) that may
# hang indefinitely on:
#   - Network partitions (HTTP connection established but no response)
#   - Misbehaving local models (infinite generation loop)
#   - Provider rate-limit retries with exponential backoff
# Without a timeout, the entire hatch process can stall in Phase 2.5
# waiting for an LLM that never returns.  90s is generous for B3/B4
# prompts (≤4KB body + system prompt → ~2K tokens input, <500 tokens
# output) while still bounded.
#
# Implementation note: we use a daemon thread + ``Queue`` instead of
# ``concurrent.futures.ThreadPoolExecutor`` because the latter creates
# a thread *pool* (one extra idle thread persisting for the process
# lifetime) — overkill for a single-call timeout.  The queue approach
# is the documented Python pattern for "run callable with timeout".
_KB_LLM_CALL_TIMEOUT_S: int = 90


def _call_chat_with_timeout(
    chat_fn: Callable[[str, str], str],
    system_prompt: str,
    user_prompt: str,
    *,
    timeout_s: int = _KB_LLM_CALL_TIMEOUT_S,
    stage: str = "B3/B4",
) -> str:
    """Call ``chat_fn`` with a hard timeout (v1.0.1 R3-M14).

    Runs the LLM call in a daemon thread and waits up to ``timeout_s``
    seconds.  On timeout, returns an empty string (so the caller's
    JSON parser hits ``json.loads("")`` and falls back to defaults,
    which is the existing error path).  On exception, re-raises so
    the caller's ``except Exception`` handler picks it up.

    Note: the daemon thread continues running after timeout — Python
    has no way to forcibly kill a thread.  For local LLM servers this
    means the request may still complete in the background, but the
    hatch process is unblocked and can proceed with defaults.  The
    thread is daemonic so it won't block process exit.
    """
    import queue
    import threading

    result_q: queue.Queue[tuple[str, BaseException | None]] = queue.Queue()

    def _worker() -> None:
        try:
            resp = chat_fn(system_prompt, user_prompt)
            result_q.put((resp, None))
        except BaseException as e:  # noqa: BLE001 — re-raised below
            result_q.put(("", e))

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    try:
        resp, err = result_q.get(timeout=timeout_s)
    except queue.Empty:
        logger.warning(
            "%s: LLM call timed out after %ds — using defaults",
            stage, timeout_s,
        )
        return ""

    if err is not None:
        raise err
    return resp


# v1.0.1 (R2-H1 regression + R3-M12): Coercion helpers for LLM JSON
# responses.  LLMs occasionally return:
#   - dict elements inside ``when_to_retrieve`` (e.g. ``[{"trigger": "x"}]``)
#     — previously ``str(x)`` produced ``"{'trigger': 'x'}"`` which got
#     injected into the runtime system prompt as garbage context.
#   - string ``"false"`` / ``"true"`` instead of booleans —
#     ``bool("false")`` is ``True`` (non-empty string), so citation_required
#     and enable_llm_rerank got silently inverted.
# These helpers silently drop / coerce bad elements and log a warning
# so B3/B4 output stays usable even when the LLM misbehaves.
def _coerce_str_list(raw: Any) -> list[str]:
    """Coerce an LLM JSON value into a ``list[str]``.

    Accepts a list of strings, dicts (extracts first string value), or
    other types (skipped).  Returns the cleaned list; logs a warning
    per dropped element type so users can spot misbehaving LLM output.
    """
    if not isinstance(raw, list):
        return []
    result: list[str] = []
    for x in raw:
        if isinstance(x, str):
            result.append(x)
        elif isinstance(x, dict):
            # Extract the first string value (common LLM pattern).
            for v in x.values():
                if isinstance(v, str):
                    result.append(v)
                    break
        else:
            # Numbers, bools, None — coerce via str() to avoid silent
            # data loss; users can see the coercion via the warning.
            logger.warning(
                "B3 coerce_str_list: non-string element %r — coercing via str()",
                type(x).__name__,
            )
            result.append(str(x))
    return result


def _coerce_bool(val: Any, default: bool) -> bool:
    """Coerce an LLM JSON value into a ``bool``.

    Handles ``true``/``false`` (bool), ``"true"``/``"false"`` (string,
    case-insensitive), ``1``/``0`` (int).  Falls back to ``default``
    for ``None`` or unrecognized values.
    """
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        s = val.strip().lower()
        if s in ("true", "1", "yes", "y"):
            return True
        if s in ("false", "0", "no", "n", ""):
            return False
        return default
    if isinstance(val, int):
        return val != 0
    return default


def _extract_json_from_response(response: str) -> str:
    """Extract a JSON object from an LLM response (v1.0.1 R3-M13).

    LLMs are told to return "STRICT JSON only — no prose, no markdown
    fences" but routinely wrap output in ```` ```json ... ``` ````
    anyway, OR prepend prose like "Here's the JSON:" before the block.
    The previous implementation only handled the case where the
    response started with ```` ``` ```` AND ended with ```` ``` ````
    exactly — any deviation (leading whitespace, prose prefix,
    missing closing fence, mixed-case ```` ```JSON ````) caused
    ``json.loads`` to fail and silently return defaults.

    Strategy:
      1. Try the raw response as-is (fast path — correct LLM behavior).
      2. Try the content of the first ```` ```...``` ```` fenced block
         (case-insensitive language tag, allows inline whitespace).
      3. Try brace-balanced extraction — find the first ``{`` and
         scan to its matching ``}``, accounting for string literals
         and escapes.  Handles prose-prefixed JSON like
         "Here's the JSON: {...}".

    Returns the extracted text on success, or the original ``response``
    on failure (so the subsequent ``json.loads`` raises a meaningful
    ``JSONDecodeError`` logged by the caller).
    """
    import re

    if not response:
        return response

    text = response.strip()
    if not text:
        return response

    # 1. Fast path: try the whole response as JSON.
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # 2. Fenced block: ```json\n{...}\n``` or ```\n{...}\n```.
    #    Case-insensitive on the language tag, allow trailing whitespace.
    fence_match = re.search(
        r"```(?:[a-zA-Z]*)?\s*\n(.*?)\n\s*```",
        text,
        re.DOTALL,
    )
    if fence_match:
        candidate = fence_match.group(1).strip()
        try:
            json.loads(candidate)
            return candidate
        except json.JSONDecodeError:
            pass  # Fall through to brace-balanced extraction

    # 3. Brace-balanced extraction — find the first ``{`` and scan
    #    to its matching ``}``, accounting for string literals and
    #    escapes.  Handles "Here's the JSON: {...}" and similar prose
    #    prefixes that LLMs often add despite instructions.
    first_brace = text.find("{")
    if first_brace >= 0:
        depth = 0
        in_string = False
        escape = False
        for i, ch in enumerate(text[first_brace:], start=first_brace):
            if escape:
                escape = False
                continue
            if ch == "\\" and in_string:
                escape = True
                continue
            if ch == '"' and not escape:
                in_string = not in_string
                continue
            if in_string:
                continue
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[first_brace:i + 1]
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return candidate  # Let caller log the parse error
        # Unbalanced braces — return the tail and let json.loads fail loudly.
        return text[first_brace:]

    return response

# ─────────────────────────────────────────────────────────────────────────
# B2: Deterministic KB mention detector
# ─────────────────────────────────────────────────────────────────────────

# Phrases that explicitly reference an external knowledge base
_KB_REFERENCE_PHRASES: tuple[str, ...] = (
    "knowledge base",
    "knowledge-base",
    "knowledgebase",
    "reference document",
    "reference folder",
    "references/",
    "references\\",
    "docs/",
    "documentation folder",
    "vector index",
    "rag retrieval",
    "retrieve tool",
    "recall tool",
)

# Frontmatter keys that declare KB usage explicitly
_KB_FRONTMATTER_KEYS: tuple[str, ...] = (
    "knowledge_base",
    "knowledge-base",
    "knowledgebase",
    "kb",
    "references",
    "docs",
)


@dataclass
class KBDetectorResult:
    """Result of B2 ``detect_kb_mention()``."""

    mentioned_explicitly: bool
    mentions: list[str] = field(default_factory=list)
    """Evidence strings (e.g. 'frontmatter key: kb', 'body phrase: knowledge base')."""
    frontmatter_kb_config: dict[str, Any] | None = None
    """Parsed KB config from frontmatter, if ``knowledge_base`` key present."""
    kb_dir_name: str = ""
    """Directory name of the KB path — used for body-mention matching."""


def detect_kb_mention(
    raw_body: str,
    frontmatter: dict[str, Any] | None,
    kb_path: Path,
) -> KBDetectorResult:
    """B2: Deterministic scan of SKILL.md for explicit KB references.

    Args:
        raw_body: SKILL.md body (after frontmatter stripped).
        frontmatter: Parsed frontmatter dict, or None.
        kb_path: Resolved absolute path to the knowledge base directory.

    Returns:
        KBDetectorResult with ``mentioned_explicitly=True`` if any
        explicit signal is found (frontmatter KB key or body phrase).

    Note:
        We deliberately do NOT treat a body mention of the KB directory
        name as an explicit signal.  A directory named ``aetheria`` would
        match every body reference to the planet "Aetheria" — coincidence,
        not intent.  Only KB-specific vocabulary ("knowledge base",
        "reference document", "references/") counts as explicit.
    """
    mentions: list[str] = []
    frontmatter_kb_config: dict[str, Any] | None = None
    kb_dir_name = kb_path.name

    # 1. Frontmatter keys
    if frontmatter:
        for key in _KB_FRONTMATTER_KEYS:
            if key in frontmatter:
                val = frontmatter[key]
                mentions.append(f"frontmatter key: {key}")
                if key in ("knowledge_base", "knowledge-base", "knowledgebase", "kb"):
                    if isinstance(val, dict):
                        frontmatter_kb_config = val

    # 2. Body phrase scan (case-insensitive)
    body_lower = raw_body.lower()
    for phrase in _KB_REFERENCE_PHRASES:
        if phrase.lower() in body_lower:
            mentions.append(f"body phrase: {phrase}")

    return KBDetectorResult(
        mentioned_explicitly=bool(mentions),
        mentions=mentions,
        frontmatter_kb_config=frontmatter_kb_config,
        kb_dir_name=kb_dir_name,
    )


# ─────────────────────────────────────────────────────────────────────────
# B3: LLM-driven usage inferencer
# ─────────────────────────────────────────────────────────────────────────

_INFER_USAGE_SYSTEM_PROMPT = """\
You are the KB_Usage_Inferencer (Harness B3) for the agenthatch compiler.

Given a SKILL.md body and a list of knowledge-base file names, infer
*when* the agent should retrieve from the KB and *how* to integrate
retrieved context into responses.

Return STRICT JSON only — no prose, no markdown fences — matching:
{
  "when_to_retrieve": [string, ...],         // 3-6 triggers
  "query_templates": [string, ...],          // 3-5 templates with {param}
  "integration_pattern": "tool_call_then_answer" | "prepend_context" | "auto_inject_on_keyword",
  "max_results_per_query": int,              // 3-8
  "citation_required": bool,
  "fallback_when_no_match": "inform_user" | "proceed_without_kb"
}

Defaults if uncertain:
  integration_pattern = "tool_call_then_answer"
  max_results_per_query = 5
  citation_required = true
  fallback_when_no_match = "inform_user"

The agent will see the KB as a ``retrieve(query, top_k)`` tool.  Prefer
``tool_call_then_answer`` so the LLM can decide when retrieval is needed
rather than injecting context that may not be relevant.
"""


def _build_infer_usage_user_prompt(
    raw_body: str,
    kb_files: list[str],
    skill_summary: str,
) -> str:
    """Build the user message for B3 inference."""
    body_excerpt = raw_body[:_MAX_KB_SAMPLE_CHARS]
    if len(raw_body) > _MAX_KB_SAMPLE_CHARS:
        body_excerpt += "\n... (truncated)"
    files_preview = "\n".join(f"- {f}" for f in kb_files[:_MAX_KB_FILES_PREVIEW])
    if len(kb_files) > _MAX_KB_FILES_PREVIEW:
        files_preview += f"\n... ({len(kb_files) - _MAX_KB_FILES_PREVIEW} more files)"
    return (
        f"## Skill Summary\n{skill_summary}\n\n"
        f"## SKILL.md Body\n```\n{body_excerpt}\n```\n\n"
        f"## Knowledge Base Files ({len(kb_files)} total)\n{files_preview}\n\n"
        f"## Task\nInfer when this agent should call ``retrieve(query, top_k)`` "
        f"to consult the knowledge base, and how to integrate results."
    )


def _parse_infer_usage_response(response: str) -> KBUsageStrategy:
    """Parse LLM JSON response into KBUsageStrategy.

    Falls back to defaults on any parse error — never raises.
    """
    defaults = KBUsageStrategy()
    if not response or not response.strip():
        return defaults

    # v1.0.1 (R3-M13): Use the unified JSON extractor instead of a
    # bespoke ``startswith("```")`` strip.  Handles prose-prefixed JSON,
    # missing closing fences, mixed-case language tags, and unbalanced
    # braces.  Previously any of these caused silent fallback to defaults.
    text = _extract_json_from_response(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("B3 infer_kb_usage: JSON parse failed (%s) — using defaults", e)
        return defaults

    # Validate enum values; fall back to defaults if invalid
    integration = data.get("integration_pattern", "tool_call_then_answer")
    if integration not in ("tool_call_then_answer", "prepend_context", "auto_inject_on_keyword"):
        integration = "tool_call_then_answer"

    fallback = data.get("fallback_when_no_match", "inform_user")
    if fallback not in ("inform_user", "proceed_without_kb"):
        fallback = "inform_user"

    try:
        # v1.0.1: Clamp max_results_per_query to valid range (1-10).
        # The system prompt asks for 3-8 but LLMs sometimes return 0 or 100.
        max_results = int(data.get("max_results_per_query", 5))
        max_results = max(1, min(max_results, 10))

        # v1.0.1 (R2-H1): Guard against LLM returning a string instead
        # of a list for ``when_to_retrieve`` / ``query_templates``.
        # Previously ``[str(x) for x in "when user asks"]`` iterated
        # the string character-by-character, producing
        # ``['w', 'h', 'e', 'n', ...]`` which then got rendered into
        # the runtime WHEN_TO_RETRIEVE constant and injected into the
        # LLM system prompt as garbage context.
        wtr_raw = data.get("when_to_retrieve", [])
        if not isinstance(wtr_raw, list):
            logger.warning(
                "B3 infer_kb_usage: when_to_retrieve is %s, expected list — "
                "coercing to empty list",
                type(wtr_raw).__name__,
            )
            wtr_raw = []
        qt_raw = data.get("query_templates", [])
        if not isinstance(qt_raw, list):
            logger.warning(
                "B3 infer_kb_usage: query_templates is %s, expected list — "
                "coercing to empty list",
                type(qt_raw).__name__,
            )
            qt_raw = []

        return KBUsageStrategy(
            when_to_retrieve=_coerce_str_list(wtr_raw),
            query_templates=_coerce_str_list(qt_raw),
            integration_pattern=integration,
            max_results_per_query=max_results,
            # v1.0.1 (R3-M12): Use _coerce_bool to handle LLM returning
            # "false"/"true" strings — bool("false") was True (inverted).
            citation_required=_coerce_bool(
                data.get("citation_required", True), True
            ),
            fallback_when_no_match=fallback,
        )
    except (TypeError, ValueError) as e:
        logger.warning("B3 infer_kb_usage: schema construction failed (%s) — using defaults", e)
        return defaults


def infer_kb_usage(
    raw_body: str,
    kb_files: list[str],
    skill_summary: str,
    chat_fn: Callable[[str, str], str],
) -> KBUsageStrategy:
    """B3: LLM-driven inference of KB usage strategy (temp=0.3).

    Args:
        raw_body: SKILL.md raw body.
        kb_files: List of KB file names (basename only).
        skill_summary: Short summary of the skill's intent.
        chat_fn: ``(system_prompt, user_prompt) -> response_text`` callable.

    Returns:
        KBUsageStrategy populated from LLM response, or defaults on error.
    """
    user_prompt = _build_infer_usage_user_prompt(raw_body, kb_files, skill_summary)
    try:
        # v1.0.1 (R3-M14): Wrap in timeout so a hung LLM call doesn't
        # stall the entire hatch process in Phase 2.5.
        response = _call_chat_with_timeout(
            chat_fn,
            _INFER_USAGE_SYSTEM_PROMPT,
            user_prompt,
            stage="B3 infer_kb_usage",
        )
    except Exception as e:
        logger.warning("B3 infer_kb_usage: LLM call failed (%s) — using defaults", e)
        return KBUsageStrategy()
    return _parse_infer_usage_response(response)


# ─────────────────────────────────────────────────────────────────────────
# B4: LLM-driven prompt generator
# ─────────────────────────────────────────────────────────────────────────

_GENERATE_PROMPT_SYSTEM_PROMPT = """\
You are the KB_Prompt_Generator (Harness B4) for the agenthatch compiler.

Given a skill summary, KB usage strategy, and KB file list, generate
three pieces of text that will be injected into the runtime agent:

1. ``system_prompt_section`` — A 3-6 sentence block (markdown) that
   tells the LLM the KB exists, what kind of information it contains,
   and that a ``retrieve(query, top_k)`` tool is available.  Should
   instruct the LLM to retrieve before answering when the question
   relates to the KB's domain.  CRITICAL — also instruct the LLM to:
   (a) retrieve exactly ONCE per user question — a single well-crafted
       query returns up to top_k chunks, which is enough;
   (b) NOT retrieve for topics the user did not ask about — over-
       retrieving pollutes the conversation context and confuses
       subsequent answers in multi-turn dialogue;
   (c) answer immediately after the retrieve call returns, using the
       retrieved chunks as evidence;
   (d) treat EACH user message as a NEW, independent question — never
       summarize or re-list answers to previous questions; if the user
       asks about topic X, answer ONLY about X, even if the conversation
       history contains retrieved chunks about other topics;
   (e) your answer must contain ONLY the substantive answer to the
       user's question — never append a meta-summary or recap, and
       never narrate retrieval.  Forbidden patterns include:
         - Chinese: "已回答…", "完整解答了…", "用户的请求…已在前一轮
           完整回答", "无剩余步骤待执行", "先检索…", "我需要查阅…".
         - English: "The request has been fully addressed",
           "I have answered the question above", "No remaining steps",
           "Task complete", "Let me search the knowledge base",
           "I'll retrieve the relevant information first."
       Any sentence that describes what you just did or are about to
       do violates this rule.  The runtime loop returns your answer
       text verbatim, so meta-commentary becomes redundant trailing
       text the user has to read past.

2. ``retrieve_tool_description`` — A 1-3 sentence description of the
   ``retrieve`` tool for the LLM's tool-use prompt.  Must mention
   parameters ``query: str`` and ``top_k: int`` and explain what
   kind of information the tool returns.

3. ``integration_instructions`` — A 2-4 sentence block instructing the
   LLM how to use retrieved chunks: cite sources, handle no-match
   cases, avoid fabricating beyond retrieved content.

Return STRICT JSON only — no prose, no markdown fences — matching:
{
  "system_prompt_section": "string",
  "retrieve_tool_description": "string",
  "integration_instructions": "string"
}
"""


def _build_generate_prompt_user_prompt(
    skill_summary: str,
    usage: KBUsageStrategy,
    kb_files: list[str],
) -> str:
    """Build the user message for B4 prompt generation."""
    files_preview = "\n".join(f"- {f}" for f in kb_files[:_MAX_KB_FILES_PREVIEW])
    if len(kb_files) > _MAX_KB_FILES_PREVIEW:
        files_preview += f"\n... ({len(kb_files) - _MAX_KB_FILES_PREVIEW} more files)"
    when_list = "\n".join(f"- {w}" for w in usage.when_to_retrieve) or "- (no triggers inferred)"
    templates = ", ".join(usage.query_templates) or "(no templates)"
    return (
        f"## Skill Summary\n{skill_summary}\n\n"
        f"## KB Usage Strategy\n"
        f"- integration_pattern: {usage.integration_pattern}\n"
        f"- max_results_per_query: {usage.max_results_per_query}\n"
        f"- citation_required: {usage.citation_required}\n"
        f"- fallback_when_no_match: {usage.fallback_when_no_match}\n"
        f"- when_to_retrieve:\n{when_list}\n"
        f"- query_templates: {templates}\n\n"
        f"## KB Files ({len(kb_files)} total)\n{files_preview}\n\n"
        f"## Task\nGenerate the three prompt artifacts."
    )


def _parse_generate_prompt_response(response: str) -> KBPromptArtifact:
    """Parse LLM JSON response into KBPromptArtifact."""
    defaults = KBPromptArtifact()
    if not response or not response.strip():
        return defaults

    # v1.0.1 (R3-M13): Use the unified JSON extractor (see B3 above).
    text = _extract_json_from_response(response)

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("B4 generate_kb_prompt: JSON parse failed (%s) — using defaults", e)
        return defaults

    try:
        return KBPromptArtifact(
            system_prompt_section=str(data.get("system_prompt_section", "")),
            retrieve_tool_description=str(data.get("retrieve_tool_description", "")),
            integration_instructions=str(data.get("integration_instructions", "")),
        )
    except (TypeError, ValueError) as e:
        logger.warning("B4 generate_kb_prompt: schema construction failed (%s) — using defaults", e)
        return defaults


def generate_kb_prompt(
    skill_summary: str,
    usage: KBUsageStrategy,
    kb_files: list[str],
    chat_fn: Callable[[str, str], str],
) -> KBPromptArtifact:
    """B4: LLM-driven generation of KB prompt artifacts (temp=0.2).

    Args:
        skill_summary: Short summary of the skill's intent.
        usage: KBUsageStrategy from B3.
        kb_files: List of KB file names.
        chat_fn: ``(system_prompt, user_prompt) -> response_text`` callable.

    Returns:
        KBPromptArtifact populated from LLM response, or defaults on error.
    """
    user_prompt = _build_generate_prompt_user_prompt(skill_summary, usage, kb_files)
    try:
        # v1.0.1 (R3-M14): Wrap in timeout so a hung LLM call doesn't
        # stall the entire hatch process in Phase 2.5.
        response = _call_chat_with_timeout(
            chat_fn,
            _GENERATE_PROMPT_SYSTEM_PROMPT,
            user_prompt,
            stage="B4 generate_kb_prompt",
        )
    except Exception as e:
        logger.warning("B4 generate_kb_prompt: LLM call failed (%s) — using defaults", e)
        return KBPromptArtifact()
    return _parse_generate_prompt_response(response)


# ─────────────────────────────────────────────────────────────────────────
# KB file discovery
# ─────────────────────────────────────────────────────────────────────────


def discover_kb_files(
    kb_path: Path,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> list[Path]:
    """Recursively discover KB source files matching include patterns.

    Args:
        kb_path: Directory or single file to scan.
        include_patterns: Glob patterns (default: *.md, *.txt, *.rst, *.markdown).
        exclude_patterns: Glob patterns to exclude (matched against the
            path relative to ``kb_path`` and the basename).  Applied
            BEFORE returning so excluded files never reach B3/B4 LLM
            context (v1.0.1: previously excluded file names were still
            sent to the LLM, leaking their existence even though they
            were skipped at index-build time).

    Returns:
        Sorted list of file Paths.  Empty if path doesn't exist.
    """
    patterns = include_patterns or list(_DEFAULT_INCLUDE_PATTERNS)
    excludes = exclude_patterns or []
    if not kb_path.exists():
        logger.warning("KB path does not exist: %s", kb_path)
        return []

    if kb_path.is_file():
        # Single-file mode: skip exclude check (user explicitly passed
        # this file as the KB target).
        return [kb_path]

    files: list[Path] = []
    for pattern in patterns:
        files.extend(kb_path.rglob(pattern))
    # Deduplicate and sort
    seen: set[Path] = set()
    unique: list[Path] = []
    for f in sorted(files):
        if f in seen:
            continue
        seen.add(f)
        # v1.0.1: Apply exclude_patterns here (not just in engine.py's
        # index builder) so excluded files don't leak into B3/B4 LLM
        # context.  Match logic mirrors engine._matches_pattern: both
        # the relative path and the basename are tried, so users can
        # write ``private/*`` (path match) or ``*.secret`` (basename).
        if excludes:
            try:
                rel = f.relative_to(kb_path)
                rel_str = str(rel)
            except ValueError:
                rel_str = f.name
            if _matches_exclude_pattern(rel_str, excludes):
                continue
        unique.append(f)
    return unique


def _matches_exclude_pattern(path: str, patterns: list[str]) -> bool:
    """Match ``path`` against any of the glob-style ``patterns``.

    Mirrors :func:`agenthatch.generate.engine._matches_pattern` so
    include/exclude semantics stay consistent between discovery and
    index build.  Tries both the full relative path and the basename
    so users can write either ``draft/*`` (path match) or ``*.tmp``
    (basename match).
    """
    import fnmatch
    from os.path import basename
    return any(
        fnmatch.fnmatch(path, p) or fnmatch.fnmatch(basename(path), p)
        for p in patterns
    )


# ─────────────────────────────────────────────────────────────────────────
# Orchestrator: B2 → B3 → B4
# ─────────────────────────────────────────────────────────────────────────


def run_kb_pipeline(
    *,
    raw_body: str,
    frontmatter: dict[str, Any] | None,
    kb_path: Path,
    skill_id: str,
    skill_summary: str,
    chat_fn: Callable[[str, str], str] | None,
) -> KnowledgeBaseConfig:
    """Orchestrate B2 → B3 → B4 to produce a KnowledgeBaseConfig.

    Pipeline:
      1. Resolve kb_path to absolute path; populate KnowledgeBaseSource.
      2. Discover KB files (for LLM context + later indexing).
      3. B2: detect explicit KB mention in SKILL.md.
      4. If frontmatter declares explicit ``usage_strategy`` /
         ``prompt_artifact`` → use them directly and skip B3/B4
         (saves API cost, respects user-declared strategy).
         Otherwise, if chat_fn provided: B3 infer → B4 generate.
      5. Apply frontmatter overrides for index params (chunk_size, etc.).
      6. Return KnowledgeBaseConfig.

    Args:
        raw_body: SKILL.md body (after frontmatter).
        frontmatter: Parsed frontmatter dict, or None.
        kb_path: Path to knowledge base directory or file.
        skill_id: Skill identifier (for logging).
        skill_summary: Short summary of the skill's intent.
        chat_fn: LLM callable, or None to skip B3/B4 (deterministic mode).

    Returns:
        KnowledgeBaseConfig with sources, usage_strategy, prompt_artifact.
    """
    resolved = kb_path.resolve()
    # v1.0.1 (R2b-M23): Path-not-exist is a hard error, not a warning.
    # Previously the pipeline continued with an empty file list, B3/B4
    # were called with no files, and the resulting KnowledgeBaseConfig
    # was persisted to agenthatch.yaml as if KB was configured — user
    # would discover at runtime that the agent had no KB content.
    # Raising here lets the caller (hatch.py) catch and surface the error.
    if not resolved.exists():
        raise FileNotFoundError(
            f"Knowledge base path does not exist: {resolved} — "
            f"check the path passed to 'agenthatch hatch <skill> <kb>'"
        )

    # v1.0.1 (R2b-M16/M17/M22): Read include/exclude patterns from
    # frontmatter so users can scope which files get indexed.  Previously
    # the pipeline hardcoded _DEFAULT_INCLUDE_PATTERNS, ignoring any
    # frontmatter declaration like:
    #   knowledge_base:
    #     include_patterns: ["*.json"]
    # The frontmatter overrides are read LATER in this function
    # (fm_overrides), but we need them NOW to construct the source
    # correctly.  Re-extract from the detector result.
    # R2b-M17: We also reference _DEFAULT_INCLUDE_PATTERNS directly
    # instead of KnowledgeBaseSource's default to keep the single
    # source of truth in kb_pipeline.py.
    fm_cfg_early: dict[str, Any] | None = None
    if frontmatter:
        for k in ("knowledge_base", "knowledge-base", "knowledgebase", "kb"):
            v = frontmatter.get(k)
            if isinstance(v, dict):
                fm_cfg_early = v
                break
    fm_overrides_early = fm_cfg_early or {}
    raw_includes = fm_overrides_early.get(
        "include_patterns", list(_DEFAULT_INCLUDE_PATTERNS)
    )
    raw_excludes = fm_overrides_early.get("exclude_patterns", [])
    # Coerce to list[str] (frontmatter YAML may produce tuples)
    include_patterns: list[str] = (
        [str(p) for p in raw_includes] if raw_includes else list(_DEFAULT_INCLUDE_PATTERNS)
    )
    exclude_patterns: list[str] = (
        [str(p) for p in raw_excludes] if raw_excludes else []
    )

    source = KnowledgeBaseSource(
        type="local_path",
        path=str(resolved),
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )

    # v1.0.1 (R2b-M22): Pass source's include_patterns to discover_kb_files.
    # v1.0.1: Also pass exclude_patterns so excluded files never reach
    # B3/B4 LLM context.  Previously discover_kb_files ignored excludes,
    # leaking file names (e.g. ``private/secret.md``) to the LLM even
    # though engine.py skipped them at index-build time.
    kb_files = discover_kb_files(
        resolved,
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
    )
    kb_file_names = [f.name for f in kb_files]
    logger.info(
        "KB pipeline [%s]: %d files discovered at %s (patterns: %s)",
        skill_id, len(kb_files), resolved, include_patterns,
    )

    # B2: deterministic detection
    detector_result = detect_kb_mention(raw_body, frontmatter, resolved)
    if detector_result.mentioned_explicitly:
        logger.info(
            "KB pipeline [%s]: explicit mention detected (%d signals)",
            skill_id, len(detector_result.mentions),
        )
    else:
        logger.info(
            "KB pipeline [%s]: no explicit mention — will invoke LLM inference",
            skill_id,
        )

    # v1.0.1: Check frontmatter for explicit usage_strategy / prompt_artifact.
    # If the user declared these in frontmatter, use them directly and skip
    # B3/B4 LLM calls — saves API cost and respects user intent.
    explicit_usage: KBUsageStrategy | None = None
    explicit_prompt: KBPromptArtifact | None = None
    fm_cfg = detector_result.frontmatter_kb_config
    if isinstance(fm_cfg, dict):
        us_raw = fm_cfg.get("usage_strategy")
        if isinstance(us_raw, dict):
            try:
                explicit_usage = KBUsageStrategy(**us_raw)
                logger.info(
                    "KB pipeline [%s]: using frontmatter usage_strategy", skill_id
                )
            except Exception as e:
                logger.warning(
                    "KB pipeline [%s]: frontmatter usage_strategy invalid (%s) "
                    "— falling back to B3 inference", skill_id, e,
                )
        pa_raw = fm_cfg.get("prompt_artifact")
        if isinstance(pa_raw, dict):
            try:
                explicit_prompt = KBPromptArtifact(**pa_raw)
                logger.info(
                    "KB pipeline [%s]: using frontmatter prompt_artifact", skill_id
                )
            except Exception as e:
                logger.warning(
                    "KB pipeline [%s]: frontmatter prompt_artifact invalid (%s) "
                    "— falling back to B4 generation", skill_id, e,
                )

    # B3 + B4: LLM-driven inference (only if no explicit config AND chat_fn provided)
    usage = explicit_usage or KBUsageStrategy()
    prompt_artifact = explicit_prompt or KBPromptArtifact()

    need_b3 = explicit_usage is None
    need_b4 = explicit_prompt is None
    if (need_b3 or need_b4) and chat_fn is not None and kb_file_names:
        if need_b3:
            usage = infer_kb_usage(raw_body, kb_file_names, skill_summary, chat_fn)
            logger.info(
                "KB pipeline [%s]: B3 inferred %d triggers, pattern=%s",
                skill_id, len(usage.when_to_retrieve), usage.integration_pattern,
            )
        if need_b4:
            prompt_artifact = generate_kb_prompt(
                skill_summary, usage, kb_file_names, chat_fn
            )
            logger.info(
                "KB pipeline [%s]: B4 generated prompt (%d chars)",
                skill_id, len(prompt_artifact.system_prompt_section),
            )
    elif need_b3 or need_b4:
        logger.info(
            "KB pipeline [%s]: skipping LLM inference (chat_fn=%s, kb_files=%d)",
            skill_id, "provided" if chat_fn else "None", len(kb_file_names),
        )

    # v1.0.1 (H6): Apply frontmatter overrides for index params.
    # User can tune chunk_size, embedding_model, etc. via frontmatter.
    fm_overrides = fm_cfg if isinstance(fm_cfg, dict) else {}
    chunk_size = int(fm_overrides.get("chunk_size", 800))
    chunk_overlap = int(fm_overrides.get("chunk_overlap", 100))
    embedding_model = fm_overrides.get("embedding_model", "all-MiniLM-L6-v2")
    retrieval_top_k = int(fm_overrides.get("retrieval_top_k", 5))
    retrieval_alpha = float(fm_overrides.get("retrieval_alpha", 0.7))
    # v1.0.1 (C5): LLM rerank is not yet implemented — no rerank_fn is
    # ever injected at runtime.  Default False to avoid misleading users.
    # The infrastructure (set_rerank_fn, _rerank_fn check in search())
    # stays for future implementation.
    # v1.0.1 (R3-M12): Use _coerce_bool to handle frontmatter
    # ``enable_llm_rerank: "false"`` (string) — bool("false") was True.
    enable_llm_rerank = _coerce_bool(
        fm_overrides.get("enable_llm_rerank", False), False
    )

    return KnowledgeBaseConfig(
        sources=[source],
        usage_strategy=usage,
        prompt_artifact=prompt_artifact,
        # Index parameters (OpenClaw-inspired defaults, overridable via frontmatter)
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_model=embedding_model,
        retrieval_top_k=retrieval_top_k,
        retrieval_alpha=retrieval_alpha,
        enable_llm_rerank=enable_llm_rerank,
        # Build-time metadata (filled by Phase 3.5 KB builder)
        total_documents=len(kb_files),
        total_chunks=0,
        index_size_bytes=0,
    )
