"""
Metrics Calculator for Agent Trajectory Evaluation.
"""

import numpy as np
from typing import List, Dict, Any
from agent_eval.schema import TrajectoryEvaluation, AgentVariant


class AgentMetricsCalculator:
    """Aggregates dataset-wide agent trajectory metrics."""

    @staticmethod
    def calculate_variant_metrics(evaluations: List[TrajectoryEvaluation]) -> Dict[str, Any]:
        if not evaluations:
            return {"total_tasks": 0, "task_success_rate": 0.0}

        total = len(evaluations)
        completed = sum(1 for e in evaluations if e.task_completed)
        task_success_rate = completed / total

        tool_acc = [e.tool_call_accuracy for e in evaluations]
        plan_eff = [e.planning_efficiency for e in evaluations]
        arg_corr = [e.argument_correctness for e in evaluations]
        err_rec = [e.error_recovery_rate for e in evaluations]
        overall = [e.overall_score for e in evaluations]

        return {
            "total_tasks": total,
            "tasks_completed": completed,
            "task_success_rate": round(task_success_rate, 4),
            "mean_tool_call_accuracy": round(float(np.mean(tool_acc)), 4),
            "mean_planning_efficiency": round(float(np.mean(plan_eff)), 4),
            "mean_argument_correctness": round(float(np.mean(arg_corr)), 4),
            "mean_error_recovery_rate": round(float(np.mean(err_rec)), 4),
            "overall_mean_score": round(float(np.mean(overall)), 4),
        }

    @classmethod
    def calculate_suite_metrics(
        cls, all_evaluations: Dict[AgentVariant, List[TrajectoryEvaluation]]
    ) -> Dict[str, Any]:
        suite_metrics = {}
        for agent_v, evals in all_evaluations.items():
            key = agent_v.value if isinstance(agent_v, AgentVariant) else str(agent_v)
            suite_metrics[key] = cls.calculate_variant_metrics(evals)
        return suite_metrics


MetricsCalculator = AgentMetricsCalculator
