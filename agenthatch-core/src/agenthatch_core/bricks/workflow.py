"""CompiledWorkflow — structured step execution from agenthatch.yaml.

v0.7.15: Moved from template inline definition to shared agenthatch-core module
so every generated agent imports rather than duplicates these classes.

v0.9.7: Added loop_steps support — after the linear workflow completes,
the agent loops back to a user-specified step index for continued
interactive guidance (e.g. browser agents loop back to step 2
after step 7, keeping navigation/interaction guidance alive).
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

    v0.9.7: After the final step, if loop_steps is set, the workflow
    resets to loop_steps instead of returning is_complete()=True.
    This keeps operational guidance alive for interactive agents.

    Instance attributes (PEP 526 class-level annotations for dataclass-like clarity,
    actual storage is instance-level in __init__):
    """

    steps: list[WorkflowStep]
    _current_index: int

    def __init__(self, steps: list[WorkflowStep], loop_steps: int | None = None):
        self.steps = steps
        self._current_index = 0
        self._loop_steps = loop_steps

    def next_step(self) -> WorkflowStep | None:
        """Return the next step, or None when complete.

        v0.9.7: If _loop_steps is set and we've reached the end,
        reset _current_index to _loop_steps and return that step.
        """
        if self._current_index >= len(self.steps):
            if self._loop_steps is not None and 0 <= self._loop_steps < len(self.steps):
                self._current_index = self._loop_steps
                step = self.steps[self._current_index]
                self._current_index += 1
                return step
            return None
        step = self.steps[self._current_index]
        self._current_index += 1
        return step

    def is_complete(self) -> bool:
        if self._loop_steps is not None:
            return False
        return self._current_index >= len(self.steps)

    def remaining(self) -> int:
        return max(0, len(self.steps) - self._current_index)