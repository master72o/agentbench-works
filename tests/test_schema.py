"""
Tests for Schema module in Agent Trajectory Evaluation Framework.
"""

from agent_eval.schema import AgentTask, AgentVariant, TrajectoryStep, ToolCallStatus


def test_trajectory_step_parsing():
    data = {
        "step_number": 1,
        "thought": "Thinking",
        "tool_name": "search",
        "tool_args": {"q": "python"},
        "tool_output": "results",
        "status": "Success",
    }
    step = TrajectoryStep.from_dict(data)
    assert step.step_number == 1
    assert step.tool_name == "search"
    assert step.status == ToolCallStatus.SUCCESS


def test_agent_variant_enum():
    assert AgentVariant.AGENT_V1_REACT_BASELINE.value == "Agent_v1_ReAct_Baseline"
