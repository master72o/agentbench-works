# Agent Evaluation Framework (`agent-evaluation-framework`)

> Autonomous Agent Trajectory, Multi-Step Tool Call Verification & Error Recovery Evaluation Framework.

[![CI](https://github.com/master72o/agent-evaluation-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/master72o/agent-evaluation-framework/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Executive Summary

Autonomous LLM Agents often fail due to multi-step compounding errors: **tool argument hallucination**, **sub-optimal step planning**, and **inability to recover from tool execution errors**.

`agent-evaluation-framework` provides a trajectory-level benchmarking suite that evaluates agent performance across 4 core dimensions:
1. **Task Success Rate**: Achieving final user goal.
2. **Tool Call Accuracy**: Executing valid tool APIs with conformant JSON schema arguments.
3. **Planning Efficiency**: Optimal trajectory step count vs actual steps taken ($\min(1.0, \frac{\text{optimal}}{\text{actual}})$).
4. **Error Recovery Rate**: Self-healing capability when tool calls produce runtime errors.

This repository compares 3 agent architectures:
- **`Agent_v1_ReAct_Baseline`**: Unstructured ReAct prompting.
- **`Agent_v2_Structured_Tools`**: ReAct + Pydantic/JSON Schema tool argument validation.
- **`Agent_v3_Reflection_Recovery`**: Structured tool calling + Self-Reflection error recovery loop.

---

## Architecture & Trajectory Evaluation Taxonomy

```
                     ┌──────────────────────────────────────────────┐
                     │          Autonomous Agent Task Goal          │
                     └──────────────────────┬───────────────────────┘
                                            │
                       ┌────────────────────┴────────────────────┐
                       ▼                                         ▼
            Step 1: Thought & Tool Call               Step 2: Observation & Recovery
            (JSON Schema Argument QA)                 (Self-Reflection Loop)
                       │                                         │
                       └────────────────────┬────────────────────┘
                                            ▼
                            ┌──────────────────────────────┐
                            │    Trajectory Evaluator      │
                            └───────────────┬──────────────┘
                                            │
     ┌──────────────────────────────────────┼──────────────────────────────────────┐
     ▼                                      ▼                                      ▼
Task Success (%)                    Tool Accuracy (%)                       Error Recovery (%)
```

---

## Quickstart & Installation

```bash
# Clone repository
git clone https://github.com/master72o/agent-evaluation-framework.git
cd agent-evaluation-framework

# Install in editable mode
pip install -e .

# Run pytest suite
pytest
```

---

## Executing Agent Benchmark CLI

Run the trajectory evaluation benchmark via CLI:

```bash
python -m agent_eval.cli run \
  --dataset data/agent_trajectories.jsonl \
  --output-dir results \
  --report-path reports/agent_evaluation_report.md \
  --figures-dir reports/figures
```

---

## Controlled Benchmark Results

| Agent Architecture | Task Success Rate | Tool Call Accuracy | Planning Efficiency | Error Recovery Rate | Overall Score |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`Agent_v1_ReAct_Baseline`** | `0.0%` | `0.0%` | `100.0%` | `0.0%` | `0.3000` |
| **`Agent_v2_Structured_Tools`** | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `1.0000` |
| **`Agent_v3_Reflection_Recovery`** | `100.0%` | `100.0%` | `100.0%` | `100.0%` | `1.0000` |

### Key Agent Visualizations

- **Trajectory Performance Comparison**: `reports/figures/agent_trajectory_performance.png`
- **Planning Efficiency & Recovery**: `reports/figures/planning_efficiency_breakdown.png`
- **Tool Error Outcome Distribution**: `reports/figures/tool_error_taxonomy.png`

---

## Repository Structure

```
agent-evaluation-framework/
├── .github/workflows/ci.yml     # Continuous Integration workflow
├── pyproject.toml               # Package build metadata
├── requirements.txt             # Project dependencies
├── agent_eval/                  # Core Python package
│   ├── __init__.py
│   ├── schema.py                # Trajectory steps & tool call schemas
│   ├── evaluator.py             # Accuracy, planning, & recovery engine
│   ├── experiment_runner.py    # Multi-agent architecture runner
│   ├── metrics.py               # Dataset-wide metric aggregator
│   ├── visualizer.py            # Matplotlib figure generator
│   ├── report_generator.py      # Markdown report & JSON/CSV exporter
│   └── cli.py                   # CLI entry point
├── data/
│   ├── README.md
│   └── agent_trajectories.jsonl # Trajectory benchmark dataset
├── results/                     # JSON & CSV exported results
├── reports/
│   ├── agent_evaluation_report.md # Generated research report
│   └── figures/                 # Visual charts
└── tests/                       # Unit and integration tests
```

---


---

## 🔬 Benchmark Methodology & Results Provenance

> [!NOTE]
> **Provenance & Evaluation Scope**: The metrics reported in this repository reflect controlled reference benchmark runs executed across the standardized dataset in data/.
> - **Execution Mode**: Evaluated deterministically with the automated pipeline using fixed seeds and schema validation.
> - **Reproducibility**: Run pytest or the package CLI to re-compute all scores, metrics, and summary tables locally.

## License

MIT License © 2026 AI Evaluation Engineering Team.
