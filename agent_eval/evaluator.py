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


class TrajectoryEfficiencyAuditor:
    """
    Audits agent execution trajectories for redundant calls, oscillation loops,
    and calculates empirical recovery latency.
    """

    @staticmethod
    def audit_tool_redundancy(trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """
        Detects repeated calls to the same tool with identical arguments.
        """
        if not trajectory:
            return {"total_steps": 0, "redundant_calls": 0, "redundancy_ratio": 0.0, "duplicate_steps": []}

        seen_calls = set()
        duplicate_steps = []

        for step in trajectory:
            call_sig = (step.tool_name, str(sorted(step.tool_args.items())) if step.tool_args else "")
            if call_sig in seen_calls:
                duplicate_steps.append(step.step_number)
            else:
                seen_calls.add(call_sig)

        redundancy_ratio = len(duplicate_steps) / len(trajectory)
        return {
            "total_steps": len(trajectory),
            "redundant_calls": len(duplicate_steps),
            "redundancy_ratio": round(redundancy_ratio, 4),
            "duplicate_steps": duplicate_steps,
        }

    @staticmethod
    def detect_oscillation_loops(trajectory: List[TrajectoryStep], max_cycle_len: int = 3) -> Dict[str, Any]:
        """
        Detects repeating oscillation cycles in tool sequences (e.g. toolA -> toolB -> toolA -> toolB).
        """
        tools = [s.tool_name for s in trajectory if s.tool_name]
        n = len(tools)
        if n < 4:
            return {"has_loop": False, "detected_cycles": []}

        detected_cycles = []
        for cycle_len in range(1, max_cycle_len + 1):
            for i in range(n - 2 * cycle_len + 1):
                pattern = tools[i : i + cycle_len]
                next_slice = tools[i + cycle_len : i + 2 * cycle_len]
                if pattern == next_slice:
                    detected_cycles.append({
                        "cycle_length": cycle_len,
                        "pattern": pattern,
                        "start_step": i + 1,
                    })

        return {
            "has_loop": len(detected_cycles) > 0,
            "detected_cycles": detected_cycles,
        }

    @staticmethod
    def calculate_recovery_latency(trajectory: List[TrajectoryStep]) -> Dict[str, Any]:
        """
        Calculates the number of trial steps taken between an error and reaching success.
        """
        if not trajectory:
            return {"total_errors": 0, "mean_recovery_latency": 0.0, "unrecovered_count": 0}

        latencies = []
        unrecovered = 0
        i = 0
        while i < len(trajectory):
            if trajectory[i].status != ToolCallStatus.SUCCESS:
                # Seek next success
                found_success = False
                for j in range(i + 1, len(trajectory)):
                    if trajectory[j].status == ToolCallStatus.SUCCESS:
                        latencies.append(j - i)
                        found_success = True
                        i = j
                        break
                if not found_success:
                    unrecovered += 1
                    break
            i += 1

        mean_lat = float(sum(latencies) / len(latencies)) if latencies else 0.0
        return {
            "total_errors": len(latencies) + unrecovered,
            "resolved_errors": len(latencies),
            "unrecovered_count": unrecovered,
            "mean_recovery_latency": round(mean_lat, 2),
            "latency_distribution": latencies,
        }

