"""
CLI Interface for Agent Trajectory Evaluation Framework (AgentBench Works).
"""

import argparse
import sys
from agent_eval.schema import AgentDataset, AgentVariant
from agent_eval.experiment_runner import ExperimentRunner
from agent_eval.evaluator import AgentCostEstimator
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
    run_parser.add_argument(
        "--cost-model",
        default="gpt-4o",
        choices=["gpt-4o", "claude-3-5-sonnet", "qwen-2.5-72b", "deepseek-chat", "default"],
        help="LLM pricing tier used for trajectory runtime token cost estimation",
    )

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

        # Compute token consumption economics
        economics = {}
        for variant, traj_attr in [
            (AgentVariant.AGENT_V1_REACT_BASELINE, "trajectory_v1"),
            (AgentVariant.AGENT_V2_STRUCTURED_TOOLS, "trajectory_v2"),
            (AgentVariant.AGENT_V3_REFLECTION_RECOVERY, "trajectory_v3"),
        ]:
            key = variant.value
            tot_p = 0
            tot_c = 0
            tot_cost = 0.0
            completed_count = sum(1 for e in all_evaluations[variant] if e.task_completed)
            for t in dataset.tasks:
                traj = getattr(t, traj_attr, [])
                cost_est = AgentCostEstimator.estimate_cost(traj, model_name=args.cost_model)
                tot_p += cost_est.prompt_tokens
                tot_c += cost_est.completion_tokens
                tot_cost += cost_est.estimated_cost_usd

            cost_per_succ = (tot_cost / completed_count) if completed_count > 0 else 0.0
            economics[key] = {
                "total_tokens": tot_p + tot_c,
                "prompt_tokens": tot_p,
                "completion_tokens": tot_c,
                "total_cost_usd": round(tot_cost, 4),
                "cost_per_success_usd": round(cost_per_succ, 4),
            }

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
        print("==========================================================================================")
        print(f"Token Consumption & Cost Economics ({args.cost_model}):")
        print("------------------------------------------------------------------------------------------")
        for agent_v, ec in economics.items():
            print(
                f"  {agent_v:32s} | Tokens: {ec['total_tokens']:6,d} "
                f"| Cost: ${ec['total_cost_usd']:.4f} "
                f"| Cost/Success: ${ec['cost_per_success_usd']:.4f}"
            )
        print("==========================================================================================\n")

        print(f"[AgentBench Works] Exporting JSON/CSV results to: {args.output_dir}")
        ReportGenerator.export_results(all_evaluations, args.output_dir, economics=economics)

        print(f"[AgentBench Works] Generating publication figures in: {args.figures_dir}")
        Visualizer.generate_all_figures(metrics, args.figures_dir, all_evaluations=all_evaluations)

        print(f"[AgentBench Works] Generating research report at: {args.report_path}")
        ReportGenerator.generate_markdown_report(metrics, args.report_path, domain_metrics=domain_metrics, economics=economics)

        print("\n[AgentBench Works] Trajectory evaluation benchmark completed successfully!")
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
