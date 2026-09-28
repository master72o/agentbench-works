"""
Tests for Trajectory Evaluator Engine.
"""

from agent_eval.schema import AgentTask, AgentVariant, TrajectoryStep, ToolCallStatus
from agent_eval.evaluator import TrajectoryEvaluator


def test_evaluate_trajectory_success():
    task = AgentTask(
        task_id="t1",
        goal="Goal",
        optimal_steps=2,
        reference_answer="Tokyo weather is 22C.",
    )
    steps = [
        TrajectoryStep(1, "Thought 1", "tool1", {"arg": 1}, "out1", ToolCallStatus.SUCCESS),
        TrajectoryStep(2, "Thought 2", "tool2", {"arg": 2}, "out2", ToolCallStatus.SUCCESS),
    ]
    res = TrajectoryEvaluator.evaluate_trajectory(task, AgentVariant.AGENT_V3_REFLECTION_RECOVERY, steps, "Tokyo weather is 22C.")
    assert res.task_completed is True
    assert res.tool_call_accuracy == 1.0
    assert res.planning_efficiency == 1.0
    assert res.overall_score == 1.0


def test_evaluate_trajectory_error_recovery():
    task = AgentTask(
        task_id="t2",
        goal="Goal",
        optimal_steps=1,
        reference_answer="Done",
    )
    steps = [
        TrajectoryStep(1, "Thought 1", "tool1", {}, "err", ToolCallStatus.INVALID_ARGUMENTS),
        TrajectoryStep(2, "Thought 2", "tool1", {"a": 1}, "Done", ToolCallStatus.SUCCESS),
    ]
    res = TrajectoryEvaluator.evaluate_trajectory(task, AgentVariant.AGENT_V3_REFLECTION_RECOVERY, steps, "Done")
    assert res.error_recovery_rate == 1.0
    assert res.task_completed is True
