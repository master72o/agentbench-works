"""
Tests for Experiment Runner and Metrics in Agent Trajectory Evaluation.
"""

from agent_eval.schema import AgentTask, AgentVariant, TrajectoryStep, ToolCallStatus, AgentDataset
from agent_eval.experiment_runner import ExperimentRunner
from agent_eval.metrics import MetricsCalculator


def test_experiment_runner():
    task = AgentTask(
        task_id="ag_001",
        goal="Goal",
        optimal_steps=1,
        reference_answer="Answer.",
        trajectory_v1=[TrajectoryStep(1, "t", "tool", {}, "err", ToolCallStatus.INVALID_ARGUMENTS)],
        trajectory_v2=[TrajectoryStep(1, "t", "tool", {"a": 1}, "Answer.", ToolCallStatus.SUCCESS)],
        trajectory_v3=[TrajectoryStep(1, "t", "tool", {"a": 1}, "Answer.", ToolCallStatus.SUCCESS)],
        final_answer_v1="Failed",
        final_answer_v2="Answer.",
        final_answer_v3="Answer.",
    )
    res_dict = ExperimentRunner.evaluate_task_across_agents(task)
    assert AgentVariant.AGENT_V1_REACT_BASELINE in res_dict
    assert res_dict[AgentVariant.AGENT_V1_REACT_BASELINE].task_completed is False
    assert res_dict[AgentVariant.AGENT_V3_REFLECTION_RECOVERY].task_completed is True


def test_metrics_calculator(tmp_path):
    jsonl = '{"task_id":"ag_001","goal":"g","optimal_steps":1,"reference_answer":"a"}\n'
    f_path = tmp_path / "dataset.jsonl"
    f_path.write_text(jsonl)

    dataset = AgentDataset.from_jsonl(str(f_path))
    all_evals = {v: [] for v in AgentVariant}
    for task in dataset.tasks:
        res = ExperimentRunner.evaluate_task_across_agents(task)
        for v, r in res.items():
            all_evals[v].append(r)

    metrics = MetricsCalculator.calculate_suite_metrics(all_evals)
    assert "Agent_v1_ReAct_Baseline" in metrics
    assert metrics["Agent_v1_ReAct_Baseline"]["total_tasks"] == 1
