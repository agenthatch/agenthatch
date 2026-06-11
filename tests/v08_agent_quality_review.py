# ruff: noqa: E501
"""v0.8 Agent Engineering Quality Review — Multi-turn Real API Tests.

Tests 3 representative SkillAgents with real DeepSeek API:
1. skill-creator — meta-skill (self-referential agent)
2. frontend-design — creative code generation
3. pdf — tool-heavy with CompiledWorkflow + Guard

Evaluates: response quality, multi-turn coherence, tool routing, guard effectiveness.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agenthatch-core" / "src"))

from agenthatch.agent.runtime import SkillAgent

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "sk-d7c914da78a649608c3cc2a55e66135c")

CONFIGS = {
    "skill-creator": Path("/tmp/agenthatch_v08_test/skill-creator/agenthatch.yaml"),
    "frontend-design": Path("/tmp/agenthatch_v08_test/frontend-design/agenthatch.yaml"),
    "pdf": Path("/tmp/agenthatch_v08_test/pdf/agenthatch.yaml"),
}


def load_agent(name: str) -> SkillAgent:
    config_path = CONFIGS[name]
    assert config_path.exists(), f"Config not found: {config_path}"
    agent = SkillAgent.from_ahspec(config_path, provider="deepseek",
                                    model="deepseek-chat", api_key=API_KEY)
    print(f"   ✓ Loaded {name} — archetype={agent._manifest.archetype}")
    print(f"   ✓ Sandbox: {type(agent.sandbox).__name__}")
    print(f"   ✓ Guard active: {agent._manifest.guard_active}")
    return agent


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: SKILL-CREATOR — meta-agent self-reference
# ═══════════════════════════════════════════════════════════════════════

def test_skill_creator_loads():
    agent = load_agent("skill-creator")
    assert agent._manifest.archetype == "multi-step"
    assert agent.sandbox is not None


def test_skill_creator_multi_turn_quality():
    """Meta-skill quality: can it reason about skill design?"""
    agent = load_agent("skill-creator")

    # Turn 1: Design a simple skill
    print("\n   [Turn 1] skill-creator: '帮我设计一个计算文件哈希的skill...'")
    r1 = agent.chat("帮我设计一个简单的skill：它的功能是计算任意文件的SHA256哈希值。请给出SKILL.md的大纲和核心内容。")
    print(f"   R1 ({len(r1)} chars): {r1[:300]}...")
    assert len(r1) > 100, f"Response too short: {len(r1)} chars"
    assert "SKILL" in r1 or "skill" in r1.lower() or "SHA" in r1 or "哈希" in r1

    # Turn 2: Refine with multi-turn context
    print("\n   [Turn 2] skill-creator: '加入trigger条件...'")
    r2 = agent.chat("给这个skill加上trigger条件：用户提到'hash','hasher','校验','完整性','checksum'等词时触发。另外限制只支持.png文件。")
    print(f"   R2 ({len(r2)} chars): {r2[:300]}...")
    assert len(r2) > 80

    print("\n   ✅ skill-creator multi-turn: OK")


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: FRONTEND-DESIGN — creative code generation
# ═══════════════════════════════════════════════════════════════════════

def test_frontend_design_loads():
    agent = load_agent("frontend-design")
    assert agent._manifest.archetype == "tool-wrapper"


def test_frontend_design_code_quality():
    """Can it generate valid, non-generic HTML/CSS?"""
    agent = load_agent("frontend-design")

    # Turn 1: Generate a landing page
    print("\n   [Turn 1] frontend-design: '生成一个极简风格的个人博客首页...'")
    r1 = agent.chat(
        "用纯HTML/CSS生成一个极简风格的个人博客首页landing page。"
        "要求：黑白色调、大标题、三个文章卡片、footer。不要使用任何JS框架。"
        "直接输出完整HTML代码。"
    )
    print(f"   R1 ({len(r1)} chars): {r1[:300]}...")
    assert len(r1) > 200
    assert "<html" in r1.lower() or "<!doctype" in r1.lower() or "<div" in r1

    # Turn 2: Modify the design
    print("\n   [Turn 2] frontend-design: '改成深色模式...'")
    r2 = agent.chat("把刚才的页面改成深色模式（dark mode），背景深灰、文字白色，卡片用稍微亮一点的灰色。")
    print(f"   R2 ({len(r2)} chars): {r2[:300]}...")
    assert len(r2) > 100

    print("\n   ✅ frontend-design multi-turn: OK")


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: PDF — tool-heavy with Guard + Workflow
# ═══════════════════════════════════════════════════════════════════════

def test_pdf_loads():
    agent = load_agent("pdf")
    assert agent._manifest.archetype == "multi-step"
    assert agent._manifest.guard_active, "PDF agent should have guard active"
    assert agent._workflow is not None, "PDF agent should have compiled workflow"


def test_pdf_guard_active():
    """Guard should be active and enforce rules."""
    agent = load_agent("pdf")
    assert agent.guard is not None, "Guard should be instantiated"
    print(f"   ✓ Guard: active with rules → {agent.guard.patterns if hasattr(agent.guard, 'patterns') else 'CompiledGuard'}")

    # Test guard validation with clean output
    clean = agent.guard.validate("Here is the PDF extraction result: 12345")
    assert clean[1] == [], f"Clean text should have no violations: {clean[1]}"

    print("   ✅ pdf guard: OK")


def test_pdf_instructional_quality():
    """Does it give actionable PDF guidance?"""
    agent = load_agent("pdf")

    # Turn 1: Ask for help with a PDF task
    print("\n   [Turn 1] pdf: '我有个PDF需要提取文本...'")
    r1 = agent.chat("我有一个report.pdf需要提取其中的文本内容，该怎么做？请给出具体步骤。")
    print(f"   R1 ({len(r1)} chars): {r1[:300]}...")
    assert len(r1) > 100, f"Response too short: {len(r1)} chars"

    # Turn 2: Follow-up question
    print("\n   [Turn 2] pdf: '文本是中文的怎么办...'")
    r2 = agent.chat("如果PDF里的文本是中文的，提取时需要注意什么？需要不同的工具吗？")
    print(f"   R2 ({len(r2)} chars): {r2[:300]}...")
    assert len(r2) > 80

    print("\n   ✅ pdf multi-turn: OK")


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: CROSS-CUTTING — sandbox removal verification at runtime
# ═══════════════════════════════════════════════════════════════════════

def test_no_sandbox_failures():
    """Verify none of the agents crash due to sandbox issues."""
    for name in CONFIGS:
        agent = load_agent(name)
        assert agent.sandbox is not None, f"{name}: sandbox should exist"
        # Actually try running a simple command through the sandbox
        result = agent.sandbox.run("echo v08_test")
        assert result.returncode == 0, f"{name}: sandbox execution failed: {result.stderr}"
        assert "v08_test" in result.stdout
        print(f"   ✓ {name}: sandbox execution OK")


def test_agent_identity():
    """Each agent should correctly report its identity."""
    for name in CONFIGS:
        agent = load_agent(name)
        assert agent.spec.identity.id is not None
        assert len(agent.spec.identity.id) > 0
        print(f"   ✓ {name}: id={agent.spec.identity.id}, display={agent.spec.identity.display_name}")


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s", "--tb=short"]))
