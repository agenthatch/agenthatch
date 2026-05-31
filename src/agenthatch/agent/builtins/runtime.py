"""Runtime builtin capabilities."""

import subprocess

from agenthatch.agent.builtins import BUILTIN_REGISTRY, BuiltinCapability


class BashRuntimeCap(BuiltinCapability):
    name = "bash_runtime"
    cap_type = "runtime"
    description = "Execute bash commands"
    schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Bash command to execute"},
        },
        "required": ["command"],
    }

    def execute(self, command: str = "") -> str:  # type: ignore[override]
        try:
            result = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: command timed out after 30s"
        except Exception as e:
            return f"Error: {e}"


class PythonRuntimeCap(BuiltinCapability):
    name = "python3_runtime"
    cap_type = "runtime"
    description = "Execute Python code"
    schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
        },
        "required": ["code"],
    }

    def execute(self, code: str = "") -> str:  # type: ignore[override]
        try:
            result = subprocess.run(
                ["python3", "-c", code],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.stdout or result.stderr or "(no output)"
        except subprocess.TimeoutExpired:
            return "Error: execution timed out after 30s"
        except Exception as e:
            return f"Error: {e}"


BUILTIN_REGISTRY["bash_runtime"] = BashRuntimeCap
BUILTIN_REGISTRY["python3_runtime"] = PythonRuntimeCap
