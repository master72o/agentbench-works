"""
Unit tests for Milestone Progress Tracking and Agent Token Cost Economics Estimator.
"""

import pytest
from agent_eval.schema import TrajectoryStep, ToolCallStatus
from agent_eval.evaluator import MilestoneAuditor, AgentCostEstimator


def test_milestone_auditor_perfect_monotonic():
    trajectory = [
        TrajectoryStep(
            step_number=1,
            thought="Step 1: Authenticate with the database",
            tool_name="auth_service",
            tool_args={"user": "admin"},
            tool_output="Authenticated session token acquired",
            status=ToolCallStatus.SUCCESS,
        ),
        TrajectoryStep(
            step_number=2,
            thought="Step 2: Query the active inventory table",
            tool_name="sql_query",
            tool_args={"table": "inventory"},
            tool_output="Inventory records retrieved: 45 items",
            status=ToolCallStatus.SUCCESS,
        ),
        TrajectoryStep(
            step_number=3,
            thought="Step 3: Export formatted CSV summary report",
            tool_name="export_csv",
            tool_args={"format": "csv"},
            tool_output="CSV summary report successfully generated",
            status=ToolCallStatus.SUCCESS,
        ),
    ]

    milestones = ["authenticate", "inventory", "summary report"]
    record = MilestoneAuditor.audit_milestones("task_db_101", trajectory, milestones)

    assert record.task_id == "task_db_101"
    assert record.total_milestones == 3
    assert record.achieved_milestones == 3
    assert record.completion_ratio == 1.0
    assert record.monotonic_progress is True
    assert record.regression_count == 0
    assert record.milestones_achieved == milestones


def test_milestone_auditor_partial_and_regression():
    trajectory = [
        TrajectoryStep(
            step_number=1,
            thought="Query inventory before authenticating",
            tool_name="sql_query",
            tool_args={"table": "inventory"},
            tool_output="Error 401: Unauthorized",
            status=ToolCallStatus.TOOL_EXECUTION_ERROR,
        ),
        TrajectoryStep(
            step_number=2,
            thought="Now authenticate session",
            tool_name="auth_service",
            tool_args={"user": "admin"},
            tool_output="Session authenticated",
            status=ToolCallStatus.SUCCESS,
        ),
    ]

    milestones = ["authenticate", "inventory", "summary report"]
    record = MilestoneAuditor.audit_milestones("task_db_102", trajectory, milestones)

    assert record.total_milestones == 3
    assert record.achieved_milestones == 2  # inventory matched in step 1, authenticate in step 2
    assert record.completion_ratio == round(2 / 3, 4)
    assert record.monotonic_progress is False
    assert record.regression_count >= 1


def test_milestone_auditor_empty_milestones():
    trajectory = [
        TrajectoryStep(
            step_number=1,
            thought="Do simple echo",
            tool_name="echo",
            tool_args={},
            tool_output="echo done",
            status=ToolCallStatus.SUCCESS,
        )
    ]
    record = MilestoneAuditor.audit_milestones("task_empty", trajectory, [])
    assert record.total_milestones == 0
    assert record.completion_ratio == 1.0
    assert record.monotonic_progress is True
    assert record.regression_count == 0


def test_agent_cost_estimator_custom_tokens():
    trajectory = [
        TrajectoryStep(
            step_number=1,
            thought="First call",
            tool_name="search",
            tool_args={},
            tool_output="results",
            prompt_tokens=500,
            completion_tokens=100,
        ),
        TrajectoryStep(
            step_number=2,
            thought="Second call",
            tool_name="analyze",
            tool_args={},
            tool_output="summary",
            prompt_tokens=1500,
            completion_tokens=200,
        ),
    ]

    # Total prompt: 2000, Total completion: 300
    # gpt-4o pricing: prompt = $0.0025 / 1k, completion = $0.010 / 1k
    # prompt_cost = (2000 / 1000) * 0.0025 = 0.0050
    # comp_cost = (300 / 1000) * 0.010 = 0.0030
    # total = 0.0080
    est = AgentCostEstimator.estimate_cost(trajectory, model_name="gpt-4o")

    assert est.model_name == "gpt-4o"
    assert est.prompt_tokens == 2000
    assert est.completion_tokens == 300
    assert est.total_tokens == 2300
    assert est.estimated_cost_usd == 0.008
    assert est.cost_per_step == 0.004


def test_agent_cost_estimator_fallback_tokens_and_model_pricing():
    trajectory = [
        TrajectoryStep(
            step_number=1,
            thought="Step without token counts",
            tool_name="ping",
            tool_args={},
            tool_output="pong",
        )
    ]

    # Fallback tokens: 250 prompt, 80 completion
    est_deepseek = AgentCostEstimator.estimate_cost(trajectory, model_name="deepseek-chat")
    assert est_deepseek.prompt_tokens == 250
    assert est_deepseek.completion_tokens == 80
    assert est_deepseek.total_tokens == 330
    assert est_deepseek.estimated_cost_usd > 0.0

    # Default model pricing
    est_default = AgentCostEstimator.estimate_cost(trajectory, model_name="unknown-model-xyz")
    assert est_default.model_name == "unknown-model-xyz"
    assert est_default.total_tokens == 330
