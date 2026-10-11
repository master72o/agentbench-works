"""
CLI Interface for Agent Trajectory Evaluation Framework (AgentBench Works).
"""

import argparse
import sys
from agent_eval.schema import AgentDataset, AgentVariant
from agent_eval.experiment_runner import ExperimentRunner
from agent_eval.metrics import MetricsCalculator
from agent_eval.visualizer import Visualizer
from agent_eval.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="AgentBench Works: Autonomous Agent Trajectory Benchmark CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    run_parser = subparsers.add_parser("run", help="Run agent trajectory benchmark")
    run_parser.add_argument("--dataset", required=True, help="Path to JSONL agent trajectory dataset")
    run_parser.add_argument("--output-dir", default="results", help="Directory to save JSON/CSV outputs")
    run_parser.add_argument("--report-path", default="reports/agent_evaluation_report.md", help="Path to save Markdown report")
    run_parser.add_argument("--figures-dir", default="reports/figures", help="Directory to save figure plots")

    args = parser.parse_args()

    if args.command == "run":
        print(f"[AgentBench Works] Loading trajectory dataset from: {args.dataset}")
        dataset = AgentDataset.from_jsonl(args.dataset)
        print(f"[AgentBench Works] Loaded {len(dataset.tasks)} multi-step agent benchmark tasks.")

        print("[AgentBench Works] Evaluating agent architectures (v1 ReAct, v2 Structured Tools, v3 Reflection & Recovery)...")
        all_evaluations = {v: [] for v in AgentVariant}

        for task in dataset.tasks:
            res_dict = ExperimentRunner.evaluate_task_across_agents(task)
            for agent_v, res in res_dict.items():
                all_evaluations[agent_v].append(res)

        metrics = MetricsCalculator.calculate_suite_metrics(all_evaluations)
        domain_metrics = MetricsCalculator.calculate_domain_metrics(all_evaluations)

        print("\n==========================================================================================")
        print("Agent Architecture Performance Summary:")
        print("==========================================================================================")
        for agent_v, m in metrics.items():
            print(
                f"  {agent_v:32s} | Success: {m['task_success_rate']*100:5.1f}% "
                f"| Tool Acc: {m['mean_tool_call_accuracy']*100:5.1f}% "
                f"| Plan Eff: {m['mean_planning_efficiency']*100:5.1f}% "
                f"| Recovery: {m['mean_error_recovery_rate']*100:5.1f}% "
                f"| Score: {m['overall_mean_score']:.4f}"
            )
        print("==========================================================================================\n")

        print(f"[AgentBench Works] Exporting JSON/CSV results to: {args.output_dir}")
        ReportGenerator.export_results(all_evaluations, args.output_dir)

        print(f"[AgentBench Works] Generating publication figures in: {args.figures_dir}")
        Visualizer.generate_all_figures(metrics, args.figures_dir, all_evaluations=all_evaluations)

        print(f"[AgentBench Works] Generating research report at: {args.report_path}")
        ReportGenerator.generate_markdown_report(metrics, args.report_path, domain_metrics=domain_metrics)

        print("\n[AgentBench Works] Trajectory evaluation benchmark completed successfully!")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
