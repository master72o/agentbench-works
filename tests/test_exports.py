"""
Unit tests for agent_eval top-level package exports.
"""

import agent_eval


def test_top_level_exports():
    expected_exports = [
        "AgentVariant",
        "ToolCallStatus",
        "TrajectoryStep",
        "AgentTask",
        "TrajectoryEvaluation",
        "AgentDataset",
        "TrajectoryEvaluator",
        "AgentMetricsCalculator",
        "ExperimentRunner",
        "ReportGenerator",
        "Visualizer",
    ]
    for exp in expected_exports:
        assert hasattr(agent_eval, exp), f"Missing export: {exp}"
        assert exp in agent_eval.__all__
