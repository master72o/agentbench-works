"""
Trajectory Evaluator Engine for Tool Call Accuracy, Planning Efficiency & Error Recovery.
"""

import re
from typing import List, Dict, Any
from agent_eval.schema import (
    AgentTask,
    AgentVariant,
    TrajectoryStep,
    ToolCallStatus,
    TrajectoryEvaluation,
)


class TrajectoryEvaluator:
    """Evaluates agent trajectory steps, tool calls, and error recovery."""

    @staticmethod
    def evaluate_trajectory(
        task: AgentTask,
        agent_version: AgentVariant,
        trajectory: List[TrajectoryStep],
        final_answer: str
    ) -> TrajectoryEvaluation:
        if not trajectory:
            return TrajectoryEvaluation(
                task_id=task.task_id,
                agent_version=agent_version,
                task_completed=False,
                tool_call_accuracy=0.0,
                planning_efficiency=0.0,
                argument_correctness=0.0,
                error_recovery_rate=0.0,
                overall_score=0.0,
            )

        total_steps = len(trajectory)

        # 1. Task Completion Check
        ans_lower = final_answer.lower()
        failure_markers = ["unable to", "could not", "failed to", "syntax error", "aborted", "does not exist"]
        is_failure_declared = any(marker in ans_lower for marker in failure_markers)

        ref_words = set(re.findall(r"\w+", task.reference_answer.lower()))
        ans_words = set(re.findall(r"\w+", ans_lower))
        if is_failure_declared:
            task_completed = False
        elif not ref_words:
            task_completed = True
        else:
            intersection = ans_words.intersection(ref_words)
            task_completed = (len(intersection) / len(ref_words)) >= 0.45

        # 2. Tool Call Accuracy
        successful_calls = sum(1 for s in trajectory if s.status == ToolCallStatus.SUCCESS)
        tool_call_accuracy = successful_calls / max(total_steps, 1)

        # 3. Planning Efficiency
        planning_efficiency = min(1.0, task.optimal_steps / max(total_steps, 1))

        # 4. Argument Correctness
        valid_args_count = sum(1 for s in trajectory if s.tool_args and s.status != ToolCallStatus.INVALID_ARGUMENTS)
        argument_correctness = valid_args_count / max(total_steps, 1)

        # 5. Error Recovery Rate
        error_steps = [i for i, s in enumerate(trajectory) if s.status != ToolCallStatus.SUCCESS]
        if not error_steps:
            error_recovery_rate = 1.0
        else:
            recovered_count = 0
            for err_idx in error_steps:
                if err_idx + 1 < total_steps and trajectory[err_idx + 1].status == ToolCallStatus.SUCCESS:
                    recovered_count += 1
            error_recovery_rate = recovered_count / len(error_steps)

        # Composite Score
        comp_score = 1.0 if task_completed else 0.0
        overall_score = (
            (comp_score * 0.35)
            + (tool_call_accuracy * 0.25)
            + (planning_efficiency * 0.20)
            + (error_recovery_rate * 0.20)
        )

        return TrajectoryEvaluation(
            task_id=task.task_id,
            agent_version=agent_version,
            task_completed=task_completed,
            tool_call_accuracy=round(tool_call_accuracy, 4),
            planning_efficiency=round(planning_efficiency, 4),
            argument_correctness=round(argument_correctness, 4),
            error_recovery_rate=round(error_recovery_rate, 4),
            overall_score=round(overall_score, 4),
            domain=getattr(task, "domain", "General"),
            difficulty=getattr(task, "difficulty", "Intermediate"),
            steps_taken=total_steps,
            optimal_steps=task.optimal_steps,
        )
