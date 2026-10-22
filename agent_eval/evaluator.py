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
    MilestoneProgressRecord,
    TrajectoryCostEstimate,
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


class MilestoneAuditor:
    """
    Audits agent execution against defined task milestones and sub-goals,
    tracking monotonic progress and identifying regression loops.
    """

    @staticmethod
    def audit_milestones(
        task_id: str,
        trajectory: List[TrajectoryStep],
        required_milestones: List[str],
    ) -> MilestoneProgressRecord:
        if not required_milestones:
            return MilestoneProgressRecord(
                task_id=task_id,
                total_milestones=0,
                achieved_milestones=0,
                completion_ratio=1.0,
                monotonic_progress=True,
                regression_count=0,
                milestones_achieved=[],
            )

        achieved: List[str] = []
        achieved_indices: List[int] = []
        regression_count = 0

        for step_idx, step in enumerate(trajectory):
            content = f"{step.thought} {step.tool_name} {step.tool_output}".lower()
            for m_idx, milestone in enumerate(required_milestones):
                if milestone.lower() in content and milestone not in achieved:
                    # Check for monotonicity: achieving milestones out of sequence
                    if achieved_indices and m_idx < max(achieved_indices):
                        regression_count += 1
                    achieved.append(milestone)
                    achieved_indices.append(m_idx)

        total = len(required_milestones)
        ratio = round(len(achieved) / total, 4) if total > 0 else 1.0
        is_monotonic = regression_count == 0 and (
            achieved_indices == sorted(achieved_indices)
        )

        return MilestoneProgressRecord(
            task_id=task_id,
            total_milestones=total,
            achieved_milestones=len(achieved),
            completion_ratio=ratio,
            monotonic_progress=is_monotonic,
            regression_count=regression_count,
            milestones_achieved=achieved,
        )


class AgentCostEstimator:
    """
    Estimates token consumption economics and USD runtime cost per trajectory
    based on standard LLM model pricing tiers.
    """

    # Pricing per 1,000 tokens (USD)
    PRICING_TABLE = {
        "gpt-4o": {"prompt": 0.0025, "completion": 0.010},
        "claude-3-5-sonnet": {"prompt": 0.0030, "completion": 0.015},
        "qwen-2.5-72b": {"prompt": 0.0004, "completion": 0.0008},
        "deepseek-chat": {"prompt": 0.00014, "completion": 0.00028},
        "default": {"prompt": 0.0015, "completion": 0.0030},
    }

    @classmethod
    def estimate_cost(
        cls,
        trajectory: List[TrajectoryStep],
        model_name: str = "default",
        fallback_prompt_tokens: int = 250,
        fallback_completion_tokens: int = 80,
    ) -> TrajectoryCostEstimate:
        rates = cls.PRICING_TABLE.get(model_name.lower(), cls.PRICING_TABLE["default"])

        total_prompt = 0
        total_completion = 0

        for step in trajectory:
            p = step.prompt_tokens if step.prompt_tokens > 0 else fallback_prompt_tokens
            c = step.completion_tokens if step.completion_tokens > 0 else fallback_completion_tokens
            total_prompt += p
            total_completion += c

        prompt_cost = (total_prompt / 1000.0) * rates["prompt"]
        comp_cost = (total_completion / 1000.0) * rates["completion"]
        total_cost = prompt_cost + comp_cost
        steps_count = max(len(trajectory), 1)

        return TrajectoryCostEstimate(
            model_name=model_name,
            prompt_tokens=total_prompt,
            completion_tokens=total_completion,
            total_tokens=total_prompt + total_completion,
            estimated_cost_usd=round(total_cost, 6),
            cost_per_step=round(total_cost / steps_count, 6),
        )


