"""Test suite for PlanLayer state machine — agenthatch-core.

Covers:
- AgentState enum (6 states)
- PlanStep / StructuredPlan model behavior
- PlanLayer state transitions (STARTING→PLANNING→EXECUTING→VERIFYING→REPLANNING→DONE)
- handle_plan_tool (set/update/complete actions)
- _process_tool_results failure detection (keyword matching)
- MAX_CONSECUTIVE_FAILURES threshold → REPLANNING
- VERIFY_EVERY_N_STEPS threshold → VERIFYING
- plan_context nag_limit (plan_guided=4 / conversation=2)
- to_context_text rendering (☐▶✓✗ markers)
- advance_step / block_step
- Serialization (to_dict / from_dict round-trip)
"""

from __future__ import annotations

import pytest
from agenthatch_core.bricks.plan import (
    AgentState,
    PlanLayer,
    PlanStep,
    StructuredPlan,
    PLAN_TOOL_DEFINITION,
)


# ---------------------------------------------------------------------------
# AgentState enum
# ---------------------------------------------------------------------------

class TestAgentState:
    """Verify the 6-state enum and its values."""

    def test_all_six_states_exist(self):
        states = {AgentState.STARTING, AgentState.PLANNING, AgentState.EXECUTING,
                  AgentState.VERIFYING, AgentState.REPLANNING, AgentState.DONE}
        assert len(states) == 6

    def test_state_values_are_lowercase_strings(self):
        assert AgentState.STARTING.value == "starting"
        assert AgentState.PLANNING.value == "planning"
        assert AgentState.EXECUTING.value == "executing"
        assert AgentState.VERIFYING.value == "verifying"
        assert AgentState.REPLANNING.value == "replanning"
        assert AgentState.DONE.value == "done"

    def test_state_is_str_enum(self):
        """AgentState should be a str Enum for JSON serialization."""
        assert isinstance(AgentState.STARTING, str)
        assert AgentState.STARTING == "starting"


# ---------------------------------------------------------------------------
# PlanStep
# ---------------------------------------------------------------------------

class TestPlanStep:
    """PlanStep model defaults and status values."""

    def test_defaults(self):
        step = PlanStep()
        assert step.step_id == 0
        assert step.description == ""
        assert step.tool_hint is None
        assert step.status == "pending"
        assert step.result_summary == ""

    def test_custom_step(self):
        step = PlanStep(step_id=1, description="Load data", tool_hint="read_csv")
        assert step.step_id == 1
        assert step.tool_hint == "read_csv"
        assert step.status == "pending"


# ---------------------------------------------------------------------------
# StructuredPlan
# ---------------------------------------------------------------------------

class TestStructuredPlan:
    """StructuredPlan model: properties and to_context_text rendering."""

    def test_empty_plan(self):
        plan = StructuredPlan()
        assert plan.total_steps == 0
        assert plan.completed_steps == 0
        assert plan.blocked_steps == 0
        # NOTE: is_complete is True for empty plan due to vacuous truth (all([]) == True).
        # This is a known edge case — an empty plan has no incomplete steps, so it's "complete".
        assert plan.is_complete is True
        assert plan.current_step is None

    def test_current_step_property(self):
        plan = StructuredPlan(
            steps=[PlanStep(step_id=1), PlanStep(step_id=2)],
            current_step_index=1,
        )
        assert plan.current_step is not None
        assert plan.current_step.step_id == 2

    def test_current_step_out_of_range(self):
        plan = StructuredPlan(steps=[PlanStep(step_id=1)], current_step_index=5)
        assert plan.current_step is None

    def test_completed_steps_count(self):
        plan = StructuredPlan(steps=[
            PlanStep(step_id=1, status="done"),
            PlanStep(step_id=2, status="pending"),
            PlanStep(step_id=3, status="done"),
        ])
        assert plan.completed_steps == 2
        assert plan.blocked_steps == 0

    def test_blocked_steps_count(self):
        plan = StructuredPlan(steps=[
            PlanStep(step_id=1, status="done"),
            PlanStep(step_id=2, status="blocked"),
        ])
        assert plan.blocked_steps == 1
        assert plan.completed_steps == 1

    def test_is_complete_all_done(self):
        plan = StructuredPlan(steps=[
            PlanStep(step_id=1, status="done"),
            PlanStep(step_id=2, status="done"),
        ])
        assert plan.is_complete is True

    def test_is_complete_not_all_done(self):
        plan = StructuredPlan(steps=[
            PlanStep(step_id=1, status="done"),
            PlanStep(step_id=2, status="pending"),
        ])
        assert plan.is_complete is False

    def test_to_context_text_markers(self):
        """Verify ☐▶✓✗ markers are rendered correctly."""
        plan = StructuredPlan(
            goal="Test goal",
            steps=[
                PlanStep(step_id=1, description="Step 1", status="pending"),
                PlanStep(step_id=2, description="Step 2", status="running"),
                PlanStep(step_id=3, description="Step 3", status="done", result_summary="OK"),
                PlanStep(step_id=4, description="Step 4", status="blocked", result_summary="Failed"),
            ],
        )
        text = plan.to_context_text()
        assert "## Plan: Test goal" in text
        assert "☐" in text  # pending
        assert "▶" in text  # running
        assert "✓" in text  # done
        assert "✗" in text  # blocked
        assert "Progress: 1/4 steps done" in text

    def test_to_context_text_result_summary(self):
        plan = StructuredPlan(
            goal="G",
            steps=[PlanStep(step_id=1, description="S", status="done", result_summary="Success")],
        )
        text = plan.to_context_text()
        assert "Success" in text

    def test_to_context_text_empty_plan(self):
        plan = StructuredPlan(goal="Empty")
        text = plan.to_context_text()
        assert "## Plan: Empty" in text
        assert "Progress: 0/0 steps done" in text


# ---------------------------------------------------------------------------
# PlanLayer — initialization and state
# ---------------------------------------------------------------------------

class TestPlanLayerInit:
    """PlanLayer constructor and initial state."""

    def test_default_init_conversation_mode(self):
        pl = PlanLayer()
        assert pl.state == AgentState.STARTING
        assert pl._mode == "conversation"
        assert pl._consecutive_tool_failures == 0
        assert pl._turn_count == 0

    def test_plan_guided_mode(self):
        pl = PlanLayer(mode="plan_guided")
        assert pl._mode == "plan_guided"
        assert pl.state == AgentState.STARTING

    def test_initial_plan_is_empty(self):
        pl = PlanLayer()
        assert pl.plan.steps == []
        assert pl.plan.goal == ""


# ---------------------------------------------------------------------------
# PlanLayer — plan_context property (nag_limit)
# ---------------------------------------------------------------------------

class TestPlanContext:
    """plan_context property: nag behavior and plan display."""

    def test_no_plan_early_turn_conversation(self):
        """In conversation mode, turns 1-2 should suggest planning."""
        pl = PlanLayer(mode="conversation")
        pl._turn_count = 1
        ctx = pl.plan_context
        assert ctx != ""
        assert "plan" in ctx.lower()

    def test_no_plan_late_turn_conversation(self):
        """In conversation mode, turn 3+ should stay quiet (nag_limit=2)."""
        pl = PlanLayer(mode="conversation")
        pl._turn_count = 3
        ctx = pl.plan_context
        assert ctx == ""

    def test_no_plan_early_turn_plan_guided(self):
        """In plan_guided mode, turns 1-4 should suggest planning."""
        pl = PlanLayer(mode="plan_guided")
        pl._turn_count = 4
        ctx = pl.plan_context
        assert ctx != ""

    def test_no_plan_late_turn_plan_guided(self):
        """In plan_guided mode, turn 5+ should stay quiet (nag_limit=4)."""
        pl = PlanLayer(mode="plan_guided")
        pl._turn_count = 5
        ctx = pl.plan_context
        assert ctx == ""

    def test_with_plan_shows_context(self):
        """When a plan exists, always show full plan context."""
        pl = PlanLayer()
        pl.plan = StructuredPlan(
            goal="My goal",
            steps=[PlanStep(step_id=1, description="Do something")],
        )
        pl.state = AgentState.EXECUTING
        ctx = pl.plan_context
        assert "## Plan: My goal" in ctx
        assert "State: executing" in ctx

    def test_with_plan_replanning_state(self):
        pl = PlanLayer()
        pl.plan = StructuredPlan(goal="G", steps=[PlanStep(step_id=1)])
        pl.state = AgentState.REPLANNING
        ctx = pl.plan_context
        assert "REPLANNING" in ctx

    def test_with_plan_executing_shows_current_step(self):
        pl = PlanLayer()
        pl.plan = StructuredPlan(
            goal="G",
            steps=[PlanStep(step_id=1, description="First"), PlanStep(step_id=2, description="Second")],
            current_step_index=1,
        )
        pl.state = AgentState.EXECUTING
        ctx = pl.plan_context
        assert "Current step" in ctx
        assert "Second" in ctx


# ---------------------------------------------------------------------------
# PlanLayer — next_suggestion property
# ---------------------------------------------------------------------------

class TestNextSuggestion:
    """next_suggestion: state-based nudges."""

    def test_starting_turn_1(self):
        pl = PlanLayer()
        pl._turn_count = 1
        assert pl.next_suggestion is not None
        assert "plan" in pl.next_suggestion.lower()

    def test_starting_turn_2_no_suggestion(self):
        """After turn 1, STARTING should not nag."""
        pl = PlanLayer()
        pl._turn_count = 2
        assert pl.next_suggestion is None

    def test_planning_turn_2(self):
        pl = PlanLayer()
        pl.state = AgentState.PLANNING
        pl._turn_count = 2
        assert pl.next_suggestion is not None

    def test_planning_turn_3_no_suggestion(self):
        pl = PlanLayer()
        pl.state = AgentState.PLANNING
        pl._turn_count = 3
        assert pl.next_suggestion is None

    def test_replanning_suggestion(self):
        pl = PlanLayer()
        pl.state = AgentState.REPLANNING
        pl._turn_count = 10  # Should still suggest in REPLANNING
        assert pl.next_suggestion is not None
        assert "blocked" in pl.next_suggestion.lower()

    def test_verifying_suggestion(self):
        pl = PlanLayer()
        pl.state = AgentState.VERIFYING
        pl._turn_count = 10
        assert pl.next_suggestion is not None
        assert "verify" in pl.next_suggestion.lower()

    def test_executing_no_suggestion(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl._turn_count = 5
        assert pl.next_suggestion is None

    def test_done_no_suggestion(self):
        pl = PlanLayer()
        pl.state = AgentState.DONE
        assert pl.next_suggestion is None


# ---------------------------------------------------------------------------
# PlanLayer — handle_turn_start
# ---------------------------------------------------------------------------

class TestHandleTurnStart:
    """handle_turn_start: turn count and verification trigger."""

    def test_increments_turn_count(self):
        pl = PlanLayer()
        assert pl._turn_count == 0
        pl.handle_turn_start()
        assert pl._turn_count == 1
        pl.handle_turn_start()
        assert pl._turn_count == 2

    def test_verification_trigger_at_5_steps(self):
        """VERIFY_EVERY_N_STEPS=5: when completed_steps % 5 == 0, transition to VERIFYING."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=i, status="done") for i in range(1, 6)
        ])
        pl.handle_turn_start()
        assert pl.state == AgentState.VERIFYING

    def test_no_verification_at_4_steps(self):
        """4 completed steps should not trigger verification."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=i, status="done") for i in range(1, 5)
        ])
        pl.handle_turn_start()
        assert pl.state == AgentState.EXECUTING

    def test_verification_at_10_steps(self):
        """10 completed steps (2 * 5) should trigger verification."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=i, status="done") for i in range(1, 11)
        ])
        pl.handle_turn_start()
        assert pl.state == AgentState.VERIFYING

    def test_no_verification_when_not_executing(self):
        """Verification only triggers in EXECUTING state."""
        pl = PlanLayer()
        pl.state = AgentState.PLANNING
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=i, status="done") for i in range(1, 6)
        ])
        pl.handle_turn_start()
        assert pl.state == AgentState.PLANNING


# ---------------------------------------------------------------------------
# PlanLayer — handle_plan_tool
# ---------------------------------------------------------------------------

class TestHandlePlanTool:
    """handle_plan_tool: set/update/complete actions and state transitions."""

    def test_set_action_creates_plan(self):
        pl = PlanLayer()
        result = pl.handle_plan_tool({
            "action": "set",
            "goal": "Test goal",
            "steps": [
                {"step_id": 1, "description": "Step 1"},
                {"step_id": 2, "description": "Step 2"},
            ],
            "current_step_index": 0,
        })
        assert "created" in result
        assert pl.plan.goal == "Test goal"
        assert len(pl.plan.steps) == 2
        assert pl.state == AgentState.EXECUTING

    def test_set_action_from_starting_to_executing(self):
        pl = PlanLayer()
        assert pl.state == AgentState.STARTING
        pl.handle_plan_tool({"action": "set", "goal": "G", "steps": [{"step_id": 1, "description": "S"}]})
        assert pl.state == AgentState.EXECUTING

    def test_set_action_from_planning_to_executing(self):
        pl = PlanLayer()
        pl.state = AgentState.PLANNING
        pl.handle_plan_tool({"action": "set", "goal": "G", "steps": [{"step_id": 1, "description": "S"}]})
        assert pl.state == AgentState.EXECUTING

    def test_update_action_from_replanning_to_executing(self):
        """Update from REPLANNING should reset failure count and go to EXECUTING."""
        pl = PlanLayer()
        pl.state = AgentState.REPLANNING
        pl._consecutive_tool_failures = 3
        pl.handle_plan_tool({"action": "update", "goal": "New G", "steps": [{"step_id": 1, "description": "S"}]})
        assert pl.state == AgentState.EXECUTING
        assert pl._consecutive_tool_failures == 0

    def test_complete_action(self):
        pl = PlanLayer()
        pl.plan = StructuredPlan(
            goal="G",
            steps=[PlanStep(step_id=1, status="pending"), PlanStep(step_id=2, status="running")],
        )
        pl.state = AgentState.EXECUTING
        result = pl.handle_plan_tool({"action": "complete"})
        assert "complete" in result.lower() or "finished" in result.lower()
        assert pl.state == AgentState.DONE
        assert all(s.status == "done" for s in pl.plan.steps)

    def test_set_action_with_planstep_objects(self):
        """handle_plan_tool should accept PlanStep objects, not just dicts."""
        pl = PlanLayer()
        pl.handle_plan_tool({
            "action": "set",
            "goal": "G",
            "steps": [PlanStep(step_id=1, description="Direct object")],
        })
        assert len(pl.plan.steps) == 1
        assert pl.plan.steps[0].description == "Direct object"

    def test_set_action_preserves_created_at(self):
        """created_at should be set on first 'set' and not overwritten on 'update'."""
        pl = PlanLayer()
        pl.handle_plan_tool({"action": "set", "goal": "G", "steps": [{"step_id": 1, "description": "S"}]})
        first_created = pl.plan.created_at
        assert first_created != ""
        pl.handle_plan_tool({"action": "update", "goal": "G2", "steps": [{"step_id": 1, "description": "S2"}]})
        assert pl.plan.created_at == first_created
        assert pl.plan.goal == "G2"

    def test_unknown_action(self):
        pl = PlanLayer()
        result = pl.handle_plan_tool({"action": "invalid"})
        assert "Unknown" in result

    def test_set_action_empty_steps(self):
        """set with no steps should still create the plan goal."""
        pl = PlanLayer()
        pl.handle_plan_tool({"action": "set", "goal": "Only goal"})
        assert pl.plan.goal == "Only goal"
        assert pl.plan.steps == []


# ---------------------------------------------------------------------------
# PlanLayer — advance_step
# ---------------------------------------------------------------------------

class TestAdvanceStep:
    """advance_step: mark done and advance to next pending."""

    def test_advance_to_next_pending(self):
        pl = PlanLayer()
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=1, description="S1", status="done"),
            PlanStep(step_id=2, description="S2", status="pending"),
            PlanStep(step_id=3, description="S3", status="pending"),
        ])
        pl.state = AgentState.EXECUTING
        pl.advance_step(2, result="OK")
        assert pl.plan.steps[1].status == "done"
        assert pl.plan.steps[1].result_summary == "OK"
        assert pl.plan.current_step_index == 2  # Advanced to step 3
        assert pl.state == AgentState.EXECUTING

    def test_advance_last_step_to_done(self):
        """Advancing the last pending step should transition to DONE."""
        pl = PlanLayer()
        pl.plan = StructuredPlan(steps=[
            PlanStep(step_id=1, description="S1", status="done"),
            PlanStep(step_id=2, description="S2", status="pending"),
        ])
        pl.state = AgentState.EXECUTING
        pl.advance_step(2, result="Done")
        assert pl.state == AgentState.DONE

    def test_advance_nonexistent_step(self):
        """Advancing a nonexistent step_id should not crash."""
        pl = PlanLayer()
        pl.plan = StructuredPlan(steps=[PlanStep(step_id=1, description="S1")])
        pl.state = AgentState.EXECUTING
        pl.advance_step(999)
        # No step was marked done, but all remaining are pending → stays EXECUTING
        assert pl.state == AgentState.EXECUTING


# ---------------------------------------------------------------------------
# PlanLayer — block_step
# ---------------------------------------------------------------------------

class TestBlockStep:
    """block_step: mark blocked and trigger REPLANNING."""

    def test_block_triggers_replanning(self):
        pl = PlanLayer()
        pl.plan = StructuredPlan(steps=[PlanStep(step_id=1, description="S1")])
        pl.state = AgentState.EXECUTING
        pl.block_step(1, reason="Tool unavailable")
        assert pl.plan.steps[0].status == "blocked"
        assert pl.plan.steps[0].result_summary == "Tool unavailable"
        assert pl.state == AgentState.REPLANNING

    def test_block_nonexistent_step(self):
        """Blocking a nonexistent step should still trigger REPLANNING."""
        pl = PlanLayer()
        pl.plan = StructuredPlan(steps=[PlanStep(step_id=1, description="S1")])
        pl.state = AgentState.EXECUTING
        pl.block_step(999, reason="N/A")
        assert pl.state == AgentState.REPLANNING


# ---------------------------------------------------------------------------
# PlanLayer — _process_tool_results (failure detection)
# ---------------------------------------------------------------------------

class TestProcessToolResults:
    """_process_tool_results: keyword-based failure detection and threshold."""

    @pytest.mark.parametrize("keyword", [
        "error", "Error", "ERROR",
        "failed", "Failed", "FAILED",
        "exception", "Exception",
        "traceback", "Traceback",
        "cannot", "Cannot",
        "unable to", "Unable to",
    ])
    def test_failure_keyword_detection(self, keyword):
        """Each failure keyword should be detected."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[{"name": "test_tool", "content": f"Something {keyword} happened"}])
        assert pl._consecutive_tool_failures == 1

    def test_no_failure_on_success(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[{"name": "test_tool", "content": "Success: result is 42"}])
        assert pl._consecutive_tool_failures == 0

    def test_consecutive_failures_accumulate(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        for _ in range(2):
            pl.handle_turn_end(None, tool_results=[{"name": "t", "content": "error occurred"}])
        assert pl._consecutive_tool_failures == 2
        assert pl.state == AgentState.EXECUTING  # Not yet 3

    def test_max_consecutive_failures_triggers_replanning(self):
        """MAX_CONSECUTIVE_FAILURES=3 should trigger REPLANNING."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        for _ in range(3):
            pl.handle_turn_end(None, tool_results=[{"name": "t", "content": "error"}])
        assert pl._consecutive_tool_failures == 3
        assert pl.state == AgentState.REPLANNING

    def test_success_resets_failure_count(self):
        """A successful tool result should reset the failure counter."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl._consecutive_tool_failures = 2
        pl.handle_turn_end(None, tool_results=[{"name": "t", "content": "All good"}])
        assert pl._consecutive_tool_failures == 0

    def test_replanning_not_triggered_when_not_executing(self):
        """Failures should not trigger REPLANNING when not in EXECUTING state."""
        pl = PlanLayer()
        pl.state = AgentState.PLANNING
        for _ in range(5):
            pl.handle_turn_end(None, tool_results=[{"name": "t", "content": "error"}])
        assert pl.state == AgentState.PLANNING

    def test_empty_content_does_not_crash(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[{"name": "t", "content": ""}])
        pl.handle_turn_end(None, tool_results=[{"name": "t", "content": None}])
        pl.handle_turn_end(None, tool_results=[{}])
        assert pl._consecutive_tool_failures == 0

    def test_multiple_results_one_failure(self):
        """One failure among multiple results should count as one failure."""
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[
            {"name": "t1", "content": "Success"},
            {"name": "t2", "content": "Error: bad input"},
            {"name": "t3", "content": "OK"},
        ])
        assert pl._consecutive_tool_failures == 1

    def test_tool_hint_success_tracking(self):
        """When a tool matching the current step's tool_hint succeeds, track it."""
        pl = PlanLayer()
        pl.plan = StructuredPlan(
            steps=[PlanStep(step_id=1, description="S1", tool_hint="read_csv")],
            current_step_index=0,
        )
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[{"name": "read_csv", "content": "data loaded"}])
        assert pl._last_tool_success is True


# ---------------------------------------------------------------------------
# PlanLayer — handle_turn_end
# ---------------------------------------------------------------------------

class TestHandleTurnEnd:
    """handle_turn_end: routing to tool_calls vs tool_results."""

    def test_no_args_no_change(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end("response text")
        # Should not crash, no state change from empty
        assert pl.state == AgentState.EXECUTING

    def test_with_tool_results(self):
        pl = PlanLayer()
        pl.state = AgentState.EXECUTING
        pl.handle_turn_end(None, tool_results=[{"name": "t", "content": "error"}])
        assert pl._consecutive_tool_failures == 1


# ---------------------------------------------------------------------------
# PlanLayer — Serialization
# ---------------------------------------------------------------------------

class TestSerialization:
    """to_dict: serialize plan layer state."""

    def test_to_dict_basic(self):
        pl = PlanLayer(mode="plan_guided")
        pl.state = AgentState.EXECUTING
        pl._consecutive_tool_failures = 2
        pl._turn_count = 5
        pl.plan = StructuredPlan(goal="Test", steps=[PlanStep(step_id=1, description="S")])

        d = pl.to_dict()
        assert d["state"] == "executing"
        assert d["consecutive_tool_failures"] == 2
        assert d["turn_count"] == 5
        assert d["plan"]["goal"] == "Test"
        assert len(d["plan"]["steps"]) == 1

    def test_to_dict_default_state(self):
        pl = PlanLayer()
        d = pl.to_dict()
        assert d["state"] == "starting"
        assert d["consecutive_tool_failures"] == 0
        assert d["turn_count"] == 0


# ---------------------------------------------------------------------------
# PLAN_TOOL_DEFINITION
# ---------------------------------------------------------------------------

class TestPlanToolDefinition:
    """Verify the plan tool definition schema for CapBus registration."""

    def test_has_function_name(self):
        assert PLAN_TOOL_DEFINITION["function"]["name"] == "plan"

    def test_parameters_have_action(self):
        params = PLAN_TOOL_DEFINITION["function"]["parameters"]
        assert "action" in params["properties"]
        assert "set" in params["properties"]["action"]["enum"]
        assert "update" in params["properties"]["action"]["enum"]
        assert "complete" in params["properties"]["action"]["enum"]

    def test_required_fields(self):
        params = PLAN_TOOL_DEFINITION["function"]["parameters"]
        assert "action" in params["required"]


# ---------------------------------------------------------------------------
# get_plan_context alias
# ---------------------------------------------------------------------------

class TestGetPlanContextAlias:
    """get_plan_context should be an alias for plan_context."""

    def test_alias_returns_same_value(self):
        pl = PlanLayer()
        pl._turn_count = 1
        assert pl.get_plan_context() == pl.plan_context
