"""
Unit tests for TrajectoryEfficiencyAuditor.
"""

import pytest
from agent_eval.schema import TrajectoryStep, ToolCallStatus
from agent_eval.evaluator import TrajectoryEfficiencyAuditor


def test_audit_tool_redundancy_clean():
    steps = [
        TrajectoryStep(step_number=1, thought="Find file", tool_name="search_file", tool_args={"query": "test.py"}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=2, thought="Read file", tool_name="read_file", tool_args={"path": "test.py"}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=3, thought="Run pytest", tool_name="run_tests", tool_args={"scope": "unit"}, tool_output="OK", status=ToolCallStatus.SUCCESS),
    ]
    res = TrajectoryEfficiencyAuditor.audit_tool_redundancy(steps)
    assert res["redundant_calls"] == 0
    assert res["redundancy_ratio"] == 0.0


def test_audit_tool_redundancy_duplicates():
    steps = [
        TrajectoryStep(step_number=1, thought="Check status", tool_name="check_status", tool_args={"job": 1}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=2, thought="Check status again", tool_name="check_status", tool_args={"job": 1}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=3, thought="Check status third time", tool_name="check_status", tool_args={"job": 1}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=4, thought="Execute", tool_name="execute", tool_args={"action": "run"}, tool_output="OK", status=ToolCallStatus.SUCCESS),
    ]
    res = TrajectoryEfficiencyAuditor.audit_tool_redundancy(steps)
    assert res["redundant_calls"] == 2
    assert res["duplicate_steps"] == [2, 3]
    assert res["redundancy_ratio"] == 0.5


def test_detect_oscillation_loops():
    # Oscillating pattern: tool_A -> tool_B -> tool_A -> tool_B
    steps = [
        TrajectoryStep(step_number=1, thought="A", tool_name="tool_a", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=2, thought="B", tool_name="tool_b", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=3, thought="A", tool_name="tool_a", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=4, thought="B", tool_name="tool_b", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
    ]
    loop_res = TrajectoryEfficiencyAuditor.detect_oscillation_loops(steps, max_cycle_len=2)
    assert loop_res["has_loop"] is True
    assert any(c["pattern"] == ["tool_a", "tool_b"] for c in loop_res["detected_cycles"])


def test_calculate_recovery_latency():
    # Step 1: Success, Step 2: Error, Step 3: Error, Step 4: Success -> latency 2
    steps = [
        TrajectoryStep(step_number=1, thought="Init", tool_name="tool_init", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
        TrajectoryStep(step_number=2, thought="Fail 1", tool_name="tool_api", tool_args={}, tool_output="Err", status=ToolCallStatus.INVALID_ARGUMENTS),
        TrajectoryStep(step_number=3, thought="Fail 2", tool_name="tool_api", tool_args={}, tool_output="Err", status=ToolCallStatus.TOOL_EXECUTION_ERROR),
        TrajectoryStep(step_number=4, thought="Fixed", tool_name="tool_api", tool_args={}, tool_output="OK", status=ToolCallStatus.SUCCESS),
    ]
    latency_res = TrajectoryEfficiencyAuditor.calculate_recovery_latency(steps)
    assert latency_res["total_errors"] >= 1
    assert latency_res["resolved_errors"] >= 1
    assert latency_res["mean_recovery_latency"] == 2.0
