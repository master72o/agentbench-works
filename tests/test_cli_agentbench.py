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
