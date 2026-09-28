"""
Experiment Runner executing Agent Trajectory Evaluations across Architectures.
"""

from typing import List, Dict
from agent_eval.schema import AgentTask, AgentVariant, TrajectoryEvaluation
from agent_eval.evaluator import TrajectoryEvaluator


class ExperimentRunner:
    """Runs trajectory evaluations across agent variants."""

    @staticmethod
    def evaluate_task_across_agents(task: AgentTask) -> Dict[AgentVariant, TrajectoryEvaluation]:
        evaluations = {}

        # 1. Agent v1 ReAct Baseline
        evaluations[AgentVariant.AGENT_V1_REACT_BASELINE] = TrajectoryEvaluator.evaluate_trajectory(
            task, AgentVariant.AGENT_V1_REACT_BASELINE, task.trajectory_v1, task.final_answer_v1
        )

        # 2. Agent v2 Structured Tools
        evaluations[AgentVariant.AGENT_V2_STRUCTURED_TOOLS] = TrajectoryEvaluator.evaluate_trajectory(
            task, AgentVariant.AGENT_V2_STRUCTURED_TOOLS, task.trajectory_v2, task.final_answer_v2
        )

        # 3. Agent v3 Reflection & Recovery
        evaluations[AgentVariant.AGENT_V3_REFLECTION_RECOVERY] = TrajectoryEvaluator.evaluate_trajectory(
            task, AgentVariant.AGENT_V3_REFLECTION_RECOVERY, task.trajectory_v3, task.final_answer_v3
        )

        return evaluations
