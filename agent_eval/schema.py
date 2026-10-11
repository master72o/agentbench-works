"""
Data Schemas for Agent Trajectory Evaluation Framework.
"""

from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


class AgentVariant(str, Enum):
    AGENT_V1_REACT_BASELINE = "Agent_v1_ReAct_Baseline"
    AGENT_V2_STRUCTURED_TOOLS = "Agent_v2_Structured_Tools"
    AGENT_V3_REFLECTION_RECOVERY = "Agent_v3_Reflection_Recovery"


class ToolCallStatus(str, Enum):
    SUCCESS = "Success"
    INVALID_ARGUMENTS = "Invalid_Arguments"
    TOOL_EXECUTION_ERROR = "Tool_Execution_Error"
    WRONG_TOOL_SELECTED = "Wrong_Tool_Selected"


@dataclass
class TrajectoryStep:
    __test__ = False

    step_number: int
    thought: str
    tool_name: str
    tool_args: Dict[str, Any]
    tool_output: str
    status: ToolCallStatus = ToolCallStatus.SUCCESS

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrajectoryStep":
        s_raw = data.get("status", "Success")
        try:
            st = ToolCallStatus(s_raw)
        except ValueError:
            st = ToolCallStatus.SUCCESS

        return cls(
            step_number=data.get("step_number", 1),
            thought=data.get("thought", ""),
            tool_name=data.get("tool_name", "unknown"),
            tool_args=data.get("tool_args", {}),
            tool_output=data.get("tool_output", ""),
            status=st,
        )


@dataclass
class AgentTask:
    __test__ = False

    task_id: str
    goal: str
    optimal_steps: int
    reference_answer: str
    domain: str = "General"
    difficulty: str = "Intermediate"
    failure_types_tested: List[str] = field(default_factory=list)
    trajectory_v1: List[TrajectoryStep] = field(default_factory=list)
    trajectory_v2: List[TrajectoryStep] = field(default_factory=list)
    trajectory_v3: List[TrajectoryStep] = field(default_factory=list)
    final_answer_v1: str = ""
    final_answer_v2: str = ""
    final_answer_v3: str = ""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AgentTask":
        def parse_steps(raw_list):
            return [TrajectoryStep.from_dict(s) if isinstance(s, dict) else s for s in raw_list]

        return cls(
            task_id=data["task_id"],
            goal=data["goal"],
            optimal_steps=data.get("optimal_steps", 2),
            reference_answer=data["reference_answer"],
            domain=data.get("domain", "General"),
            difficulty=data.get("difficulty", "Intermediate"),
            failure_types_tested=data.get("failure_types_tested", []),
            trajectory_v1=parse_steps(data.get("trajectory_v1", [])),
            trajectory_v2=parse_steps(data.get("trajectory_v2", [])),
            trajectory_v3=parse_steps(data.get("trajectory_v3", [])),
            final_answer_v1=data.get("final_answer_v1", ""),
            final_answer_v2=data.get("final_answer_v2", ""),
            final_answer_v3=data.get("final_answer_v3", ""),
        )


@dataclass
class TrajectoryEvaluation:
    task_id: str
    agent_version: AgentVariant
    task_completed: bool
    tool_call_accuracy: float
    planning_efficiency: float
    argument_correctness: float
    error_recovery_rate: float
    overall_score: float
    domain: str = "General"
    difficulty: str = "Intermediate"
    steps_taken: int = 0
    optimal_steps: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "agent_version": self.agent_version.value if isinstance(self.agent_version, AgentVariant) else str(self.agent_version),
            "domain": self.domain,
            "difficulty": self.difficulty,
            "steps_taken": self.steps_taken,
            "optimal_steps": self.optimal_steps,
            "task_completed": bool(self.task_completed),
            "tool_call_accuracy": round(self.tool_call_accuracy, 4),
            "planning_efficiency": round(self.planning_efficiency, 4),
            "argument_correctness": round(self.argument_correctness, 4),
            "error_recovery_rate": round(self.error_recovery_rate, 4),
            "overall_score": round(self.overall_score, 4),
        }


class AgentDataset:
    """Dataset container for agent evaluation tasks."""

    def __init__(self, tasks: List[AgentTask]):
        self.tasks = tasks

    @classmethod
    def from_jsonl(cls, file_path: str) -> "AgentDataset":
        import json
        tasks = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    tasks.append(AgentTask.from_dict(json.loads(line)))
        return cls(tasks=tasks)
