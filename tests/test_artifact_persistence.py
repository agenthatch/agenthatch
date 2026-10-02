"""v1.0.24: artifact persistence in the agent loop.

Tool results shaped like {"status": "ok", "type": "artifact",
"filename": ..., <payload>} must be written to disk under the artifacts
directory (``./artifacts`` by default, ``AGENTHATCH_ARTIFACTS_DIR``
overrides) and replaced with a compact confirmation for the LLM.
Anything else passes through untouched, and persistence failures are
logged (log channel only) with the original result returned unchanged.
"""
import json
from types import SimpleNamespace

import agenthatch_core.loop.agent_loop as loop_mod
from agenthatch_core.loop.agent_loop import ConversationLoop, _persist_tool_artifact


def _artifact(filename="piece.html", payload="<h1>art</h1>"):
    return {
        "status": "ok",
        "type": "artifact",
        "filename": filename,
        "html": payload,
        "title": "Algorithmic Philosophy: Test",
    }


class TestPersistToolArtifact:
    def test_artifact_written_and_compacted(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(tmp_path / "arts"))
        out = _persist_tool_artifact(_artifact())
        data = json.loads(out)
        saved = tmp_path / "arts" / "piece.html"
        assert data["status"] == "saved"
        assert data["filename"] == "piece.html"
        assert data["saved_to"] == str(saved)
        assert saved.read_text(encoding="utf-8") == "<h1>art</h1>"
        # the raw payload must NOT travel back into the LLM context
        assert "<h1>art</h1>" not in out

    def test_non_artifact_results_pass_through(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(tmp_path))
        cases = [
            {"status": "ok", "type": "data", "filename": "x", "html": "y"},
            "just a string",
            {"status": "error", "type": "artifact", "filename": "e", "html": "z"},
            {"status": "ok", "type": "artifact", "filename": "e.html"},
            {"status": "ok", "type": "artifact", "filename": "e.html", "html": ""},
            None,
            42,
        ]
        for original in cases:
            assert _persist_tool_artifact(original) is original
        assert list(tmp_path.iterdir()) == []

    def test_path_traversal_sanitized(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(tmp_path / "arts"))
        data = json.loads(_persist_tool_artifact(_artifact(filename="../../evil.txt")))
        assert data["filename"] == "evil.txt"
        assert (tmp_path / "arts" / "evil.txt").exists()
        assert not (tmp_path / "evil.txt").exists()

    def test_write_failure_returns_original(self, tmp_path, monkeypatch):
        # artifacts dir target is a regular file -> mkdir fails -> passthrough
        blocker = tmp_path / "blocker"
        blocker.write_text("i am a file", encoding="utf-8")
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(blocker))
        original = _artifact()
        assert _persist_tool_artifact(original) is original


class TestBusRawReturn:
    """v1.0.24: CapBus.route() must not stringify executor results.

    The old ``str()`` at the bus (and the twin str() in the python-tool
    executor closure) hid artifact dicts from the loop, so they could
    never reach disk. The loop is the only route() consumer and does its
    own stringification after artifact persistence.
    """

    def test_route_returns_executor_result_unstringified(self):
        from agenthatch_core.tools.bus import CapBus

        bus = CapBus(task_complete_enabled=False)
        bus.register(
            name="render",
            executor=lambda args: {
                "status": "ok", "type": "artifact",
                "filename": "x.html", "html": "<p>",
            },
            schema={"name": "render", "description": "d", "parameters": {"type": "object"}},
            source="user",
        )
        out = bus.route("render", {})
        assert isinstance(out, dict)
        assert out["filename"] == "x.html"


class TestExecuteToolCallsWiring:
    def test_sequential_path_persists_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(tmp_path / "arts"))
        monkeypatch.setattr(
            loop_mod, "_route_with_timeout",
            lambda bus, name, args: _artifact(filename="wired.html"),
        )
        fake_loop = SimpleNamespace(capbus=object())
        tc = SimpleNamespace(name="render_interactive_artifact", arguments={})
        entries = ConversationLoop._execute_tool_calls(fake_loop, [tc], parallel=False)
        assert entries[0]["result"].startswith('{"status": "saved"')
        assert (tmp_path / "arts" / "wired.html").exists()

    def test_parallel_path_persists_artifacts(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AGENTHATCH_ARTIFACTS_DIR", str(tmp_path / "arts"))
        monkeypatch.setattr(
            loop_mod, "_route_with_timeout",
            lambda bus, name, args: _artifact(filename="par.html"),
        )
        fake_loop = SimpleNamespace(capbus=object())
        tcs = [
            SimpleNamespace(name="t1", arguments={}),
            SimpleNamespace(name="t2", arguments={}),
        ]
        entries = ConversationLoop._execute_tool_calls(fake_loop, tcs, parallel=True)
        assert all(e["result"].startswith('{"status": "saved"') for e in entries)
        assert (tmp_path / "arts" / "par.html").exists()
