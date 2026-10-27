"""
Unit tests for CLI and benchmark orchestration in AgentBench Works.
"""

import subprocess
import sys
import os
from pathlib import Path
from agent_eval.schema import AgentDataset, AgentVariant
from agent_eval.experiment_runner import ExperimentRunner
from agent_eval.metrics import MetricsCalculator


def test_cli_execution_end_to_end(tmp_path):
    # Verify dataset loading and full run via python module
    dataset_path = Path("repos/agentbench-works/data/agent_trajectories.jsonl")
    if not dataset_path.exists():
        dataset_path = Path("data/agent_trajectories.jsonl")

    assert dataset_path.exists()
    dataset = AgentDataset.from_jsonl(str(dataset_path))
    assert len(dataset.tasks) >= 16

    all_evals = {v: [] for v in AgentVariant}
    for task in dataset.tasks:
        res = ExperimentRunner.evaluate_task_across_agents(task)
        for v, r in res.items():
            all_evals[v].append(r)

    metrics = MetricsCalculator.calculate_suite_metrics(all_evals)
    domain_metrics = MetricsCalculator.calculate_domain_metrics(all_evals)

    assert "Agent_v1_ReAct_Baseline" in metrics
    assert "Agent_v2_Structured_Tools" in metrics
    assert "Agent_v3_Reflection_Recovery" in metrics

    # Verify v3 outperforms v1 across success rate
    assert metrics["Agent_v3_Reflection_Recovery"]["task_success_rate"] >= metrics["Agent_v1_ReAct_Baseline"]["task_success_rate"]
    assert len(domain_metrics) >= 5


def test_domain_metrics_calculation():
    dataset_path = Path("repos/agentbench-works/data/agent_trajectories.jsonl")
    if not dataset_path.exists():
        dataset_path = Path("data/agent_trajectories.jsonl")

    dataset = AgentDataset.from_jsonl(str(dataset_path))
    all_evals = {v: [] for v in AgentVariant}
    for task in dataset.tasks[:5]:
        res = ExperimentRunner.evaluate_task_across_agents(task)
        for v, r in res.items():
            all_evals[v].append(r)

    domain_res = MetricsCalculator.calculate_domain_metrics(all_evals)
    assert isinstance(domain_res, dict)
    for dom, variants in domain_res.items():
        assert isinstance(dom, str)
        assert len(variants) > 0


def test_report_generation_with_economics(tmp_path):
    from agent_eval.report_generator import ReportGenerator

    metrics = {
        "Agent_v1_ReAct_Baseline": {
            "task_success_rate": 0.25,
            "mean_tool_call_accuracy": 0.646,
            "mean_planning_efficiency": 1.0,
            "mean_error_recovery_rate": 0.375,
            "overall_mean_score": 0.524,
        }
    }
    economics = {
        "Agent_v1_ReAct_Baseline": {
            "total_tokens": 9900,
            "prompt_tokens": 7500,
            "completion_tokens": 2400,
            "total_cost_usd": 0.0427,
            "cost_per_success_usd": 0.0107,
        }
    }
    report_file = tmp_path / "report.md"
    content = ReportGenerator.generate_markdown_report(
        metrics, str(report_file), economics=economics
    )
    assert report_file.exists()
    assert "Token Consumption & Economic Efficiency Breakdown" in content
    assert "$0.0427" in content

