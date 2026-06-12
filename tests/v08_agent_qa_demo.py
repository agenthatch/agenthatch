# ruff: noqa: E501
"""v0.8 Agent Quality Demo — Deep multi-turn conversations with SkillAgents.

This is NOT a pytest suite — it's a hands-on quality review tool.
Runs real conversations and prints full responses for human review.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agenthatch-core" / "src"))

from agenthatch.agent.runtime import SkillAgent

API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
HATCHED = Path("/tmp/agenthatch_v08_test")


def load(name: str) -> SkillAgent:
    path = HATCHED / name / "agenthatch.yaml"
    agent = SkillAgent.from_ahspec(path, provider="deepseek", model="deepseek-chat", api_key=API_KEY)
    print(f"[LOAD] {name} | archetype={agent._manifest.archetype} | guard={'ON' if agent._manifest.guard_active else 'OFF'} | sandbox={type(agent.sandbox).__name__}")
    return agent


def chat(agent: SkillAgent, label: str, msg: str) -> str:
    print(f"\n{'='*60}")
    print(f"[{label}] USER: {msg[:120]}...")
    print("-"*60)
    resp = agent.chat(msg)
    print(f"ASSISTANT ({len(resp)} chars):\n{resp[:1500]}")
    if len(resp) > 1500:
        print(f"... (truncated, {len(resp)} chars total)")
    print("-"*60)
    return resp


def main():
    # ─── SKILL-CREATOR ──────────────────────────────────────────
    sc = load("skill-creator")
    chat(sc, "SC-T1", "帮我设计一个简单的skill：功能是计算任意文件的SHA256哈希值。给出SKILL.md的完整内容。")
    chat(sc, "SC-T2", "给这个skill加上trigger条件：用户提到'hash','hasher','校验','checksum'时触发。另外限制只支持.png和.jpg文件。")
    chat(sc, "SC-T3", "你刚才设计的这个skill，如果用户传了一个不存在的文件路径，会发生什么？需要加什么错误处理？")

    # ─── FRONTEND-DESIGN ────────────────────────────────────────
    fd = load("frontend-design")
    chat(fd, "FD-T1", "生成一个极简风格的404错误页面。纯HTML/CSS，黑白色调，居中对齐，一个大大的'404'数字。直接给我完整代码。")
    chat(fd, "FD-T2", "把刚才的404页面改成深色模式（dark mode）。背景#1a1a1a，文字白色，404数字用渐变色。")

    # ─── PDF ────────────────────────────────────────────────────
    pdf = load("pdf")
    chat(pdf, "PDF-T1", "我有一个scan_report.pdf需要提取文本内容，但我不知道它是扫描件还是原生PDF。应该怎么判断？用什么工具？")
    chat(pdf, "PDF-T2", "如果确定是扫描件（图片型PDF），中文文本怎么提取？需要OCR吗？推荐什么工具？")

    print("\n" + "="*60)
    print("ALL DEMOS COMPLETE — Review outputs above for quality.")


if __name__ == "__main__":
    main()
