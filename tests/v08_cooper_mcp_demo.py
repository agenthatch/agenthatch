# ruff: noqa: E501
"""v0.8 MCP Agent Quality Review — deepseek-v4-pro with thinking.

Real multi-turn conversation with an MCP-enabled Agent reading a knowledge base page.
Uses mcporter as MCP bridge for tool calling.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agenthatch-core" / "src"))

from agenthatch.agent.runtime import SkillAgent

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HATCHED = Path("/tmp/agenthatch_v08_test")


def load_agent(name: str, model: str = "deepseek-chat") -> SkillAgent:
    path = HATCHED / name / "agenthatch.yaml"
    agent = SkillAgent.from_ahspec(
        path, provider="deepseek", model=model, api_key=API_KEY,
    )
    print(f"[LOAD] {name} | model={model} | archetype={agent._manifest.archetype}")
    print(f"  sandbox={type(agent.sandbox).__name__} | guard={'ON' if agent._manifest.guard_active else 'OFF'}")
    # List MCP servers
    mcp_names = [s.name for s in agent.spec.interface.mcp_servers]
    print(f"  MCP servers: {mcp_names}")
    return agent


def chat(agent: SkillAgent, label: str, msg: str) -> str:
    print(f"\n{'='*60}")
    print(f"[{label}] USER: {msg[:150]}")
    print("-"*60)
    resp = agent.chat(msg)
    print(f"ASSISTANT ({len(resp)} chars):\n{resp[:2000]}")
    if len(resp) > 2000:
        print(f"\n... (truncated, {len(resp)} chars total)")
    print("-"*60)
    return resp


def main():
    # ─── MCP Agent with deepseek-chat (baseline) ─────────────────────
    print("\n" + "="*60)
    print("PHASE 1: MCP Agent with deepseek-chat (BASELINE)")
    print("="*60)
    agent = load_agent("mcp-knowledge", model="deepseek-chat")

    chat(agent, "C1-CHAT",
         "请读取知识库页面，查看该页面的标题和主要内容是什么？请简要总结。")

    # Clean up session for next phase
    agent._checkpoint_mgr = None

    # ─── MCP Agent with deepseek-v4-pro (thinking enabled) ───────────
    print("\n" + "="*60)
    print("PHASE 2: MCP Agent with deepseek-v4-pro (DEEP THINKING)")
    print("="*60)
    agent_v4 = load_agent("mcp-knowledge", model="deepseek-v4-pro")

    chat(agent_v4, "C2-T1",
         "请读取知识库页面，我需要你分析这个页面的结构——有什么表格吗？"
         "有几级标题？主要内容分为几块？")

    chat(agent_v4, "C2-T2",
         "根据刚才读取的页面内容，请详细分析这个文档的类型和用途。"
         "如果这是一个需求文档/技术方案，请指出其中的关键模块和测试用例。"
         "如果包含API接口，请列出涉及的接口名称。")

    # ─── PHASE 3: Multi-turn context preservation ────────────────────
    print("\n" + "="*60)
    print("PHASE 3: Multi-turn with v4-pro (context preservation)")
    print("="*60)
    chat(agent_v4, "C3-T1",
         "列出这个知识库的目录结构。有哪些子目录/子页面？")

    chat(agent_v4, "C3-T2",
         "根据刚才看到的目录结构，这个知识库整体是关于什么主题的？"
         "各个子目录之间是什么关系？")

    print("\n" + "="*60)
    print("ALL MCP AGENT DEMOS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
