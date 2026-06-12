"""v0.8 Sandbox Removal Verification — Real API + Code Integrity Tests.

Verifies:
1. No ImportError — SandboxTier, Sandbox, BrickManifest all import correctly
2. Sandbox is always enabled (no _NullSandbox path)
3. SandboxWhitelist.default() returns full command set (including docker, pip)
4. AHCoreAgent can be instantiated and run a real chat via DeepSeek API
5. Generated template produces valid agent.py with no sandbox references
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agenthatch-core" / "src"))

from agenthatch_core.agent import AHCoreAgent
from agenthatch_core.bricks.manifest import BrickManifest, SandboxTier
from agenthatch_core.bricks.sandboxes import SandboxWhitelist
from agenthatch_core.sandbox.executor import Sandbox
from agenthatch_core.types import AgentIdentity


def test_sandbox_tier_still_exists():
    """v0.8: SandboxTier kept as backward-compat sentinel."""
    assert SandboxTier.NONE == "none"
    assert SandboxTier.STANDARD == "standard"
    assert SandboxTier.EXTENDED == "extended"


def test_brick_manifest_has_sandbox_field():
    """BrickManifest still carries sandbox for backward compat."""
    m = BrickManifest()
    assert m.sandbox == "none"


def test_whitelist_default_full():
    """v0.8.1: Whitelist no longer applied — SandboxWhitelist retained as compat sentinel."""
    wl = SandboxWhitelist.default()
    assert isinstance(wl.commands, set)


def test_ahcore_agent_always_uses_sandbox():
    """v0.8: Sandbox always instantiated, never _NullSandbox.
    v0.8.1: Whitelist removed — no _ALLOWED_COMMANDS check."""
    agent = AHCoreAgent(
        identity=AgentIdentity(id="test", display_name="Test", version="0.1.0"),
    )
    assert isinstance(agent.sandbox, Sandbox), (
        f"Expected Sandbox instance, got {type(agent.sandbox).__name__}"
    )


def test_sandbox_execution():
    """Direct subprocess execution works (echo hello)."""
    s = Sandbox()
    result = s.run("echo hello")
    assert result.returncode == 0
    assert "hello" in result.stdout


def test_sandbox_truncated_command():
    """v0.8.1: No whitelist — unknown commands return FileNotFoundError."""
    s = Sandbox()
    result = s.run("nonexistent_command_xyz arg1")
    assert result.returncode == 1
    assert "not found" in result.stderr


def test_real_api_chat():
    """End-to-end: AHCoreAgent chats via DeepSeek API with real API key."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        api_key = ""

    agent = AHCoreAgent(
        identity=AgentIdentity(id="test", display_name="Test Agent", version="1.0"),
        runtime_config={
            "llm": {
                "provider": "deepseek",
                "model": "deepseek-chat",
                "api_key": api_key,
                "base_url": "https://api.deepseek.com",
            }
        },
    )

    # Verify LLM initialized
    assert agent.llm is not None, "LLMClient should be initialized"

    # Simple chat test
    response = agent.chat("Say hello in exactly one word.")
    assert response is not None
    assert len(response) > 0
    print(f"\n✅ Real API response ({len(response)} chars): {response[:200]}")


def test_generated_template_no_sandbox():
    """Verify agent.py.j2 template has no sandbox-related code."""
    template_path = (
        Path(__file__).parent.parent / "src" / "agenthatch" / "generate"
        / "templates" / "agent.py.j2"
    )
    content = template_path.read_text()
    assert "SandboxTier" not in content, "Template must not reference SandboxTier"
    assert "sandbox" not in content.lower(), "Template must not reference sandbox"


def test_sandbox_configure_no_isolated_arg():
    """Sandbox.configure() works without isolated arg (no Docker)."""
    s = Sandbox()
    s.configure(runtime="python3", timeout="30s", env={"KEY": "val"})
    assert s.config.runtime == "python3"
    assert s.config.timeout == "30s"
    assert s.config.env == {"KEY": "val"}


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v", "-s"]))
