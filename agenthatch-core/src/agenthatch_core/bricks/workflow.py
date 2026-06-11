"""CompiledWorkflow — structured step execution from agenthatch.yaml.

v0.7.15: Moved from template inline definition to shared agenthatch-core module
so every generated agent imports rather than duplicates these classes.
"""

from dataclasses import dataclass


@dataclass
class WorkflowStep:
    """A single step in a linear workflow.

    Steps with scripts are executed deterministically during warmup
    (v0.7.9+).  Steps without scripts provide conversational guidance
    only.
    """

    step: int
    description: str
    script: str | None = None


class CompiledWorkflow:
    """Compiled linear workflow from agenthatch.yaml instructions.

    Enforces step order sequentially.  Script-bearing steps are executed
    once during agent initialisation (`_run_warmup_scripts`); the
    remaining steps are injected as conversational context
    (`_pre_turn_workflow`).

    Instance attributes (PEP 526 class-level annotations for dataclass-like clarity,
    actual storage is instance-level in __init__):
    """

    steps: list[WorkflowStep]
    _current_index: int

    def __init__(self, steps: list[WorkflowStep]):
        self.steps = steps
        self._current_index = 0

    def next_step(self) -> WorkflowStep | None:
        """Return the next step, or None when complete."""
        if self._current_index >= len(self.steps):
            return None
        step = self.steps[self._current_index]
        self._current_index += 1
        return step

    def is_complete(self) -> bool:
        return self._current_index >= len(self.steps)

    def remaining(self) -> int:
        return max(0, len(self.steps) - self._current_index)