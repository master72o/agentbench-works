"""
AgentBench Works: Autonomous Multi-Step Agent Trajectory Benchmark & Error Recovery Studio.
Streamlit application for visualizing agent trajectories, error cascades, and empirical benchmarks.
"""

import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from agent_eval.schema import AgentDataset, AgentVariant, ToolCallStatus
from agent_eval.experiment_runner import ExperimentRunner
from agent_eval.metrics import MetricsCalculator
from agent_eval.evaluator import TrajectoryEvaluator

# Page Configuration
st.set_page_config(
    page_title="AgentBench Works — Autonomous Agent Trajectory Observatory",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .metric-card {
        background-color: #1a1c24;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 18px;
        text-align: center;
    }
    .badge-success {
        background-color: #28a745;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-danger {
        background-color: #dc3545;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-warning {
        background-color: #ffc107;
        color: #212529;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    .badge-info {
        background-color: #17a2b8;
        color: white;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.85em;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_benchmark_data():
    dataset_candidates = [
        Path("data/agent_trajectories.jsonl"),
        Path("repos/agentbench-works/data/agent_trajectories.jsonl"),
    ]
    target_path = None
    for p in dataset_candidates:
        if p.exists():
            target_path = p
            break

    if not target_path:
        st.error("Benchmark dataset `data/agent_trajectories.jsonl` not found.")
        return None, None, None

    dataset = AgentDataset.from_jsonl(str(target_path))
    all_evals = {v: [] for v in AgentVariant}
    for task in dataset.tasks:
        res_dict = ExperimentRunner.evaluate_task_across_agents(task)
        for agent_v, res in res_dict.items():
            all_evals[agent_v].append(res)

    suite_metrics = MetricsCalculator.calculate_suite_metrics(all_evals)
    domain_metrics = MetricsCalculator.calculate_domain_metrics(all_evals)

    return dataset, suite_metrics, all_evals


dataset, suite_metrics, all_evals = load_benchmark_data()

# Header
st.title("🤖 AgentBench Works: Autonomous Multi-Step Agent Trajectory Benchmark")
st.markdown(
    "> **Empirical Multi-Step Agent Evaluation Observatory** comparing Traditional ReAct Prompting, "
    "Structured Tool Schema Contracts, and Self-Reflection Error Recovery across 16 Realistic Enterprise Tasks."
)

if not dataset:
    st.stop()

# Top Metrics Row
v1_key = AgentVariant.AGENT_V1_REACT_BASELINE.value
v2_key = AgentVariant.AGENT_V2_STRUCTURED_TOOLS.value
v3_key = AgentVariant.AGENT_V3_REFLECTION_RECOVERY.value

col1, col2, col3, col4 = st.columns(4)
with col1:
    v3_succ = suite_metrics[v3_key]["task_success_rate"] * 100
    v1_succ = suite_metrics[v1_key]["task_success_rate"] * 100
    st.metric(
        label="v3 Task Success Rate",
        value=f"{v3_succ:.1f}%",
        delta=f"+{v3_succ - v1_succ:.1f}% vs v1 ReAct",
    )
with col2:
    v3_tool = suite_metrics[v3_key]["mean_tool_call_accuracy"] * 100
    v1_tool = suite_metrics[v1_key]["mean_tool_call_accuracy"] * 100
    st.metric(
        label="v3 Tool Call Accuracy",
        value=f"{v3_tool:.1f}%",
        delta=f"+{v3_tool - v1_tool:.1f}% vs v1 ReAct",
    )
with col3:
    v3_rec = suite_metrics[v3_key]["mean_error_recovery_rate"] * 100
    v1_rec = suite_metrics[v1_key]["mean_error_recovery_rate"] * 100
    st.metric(
        label="v3 Error Recovery Rate",
        value=f"{v3_rec:.1f}%",
        delta=f"+{v3_rec - v1_rec:.1f}% vs v1 ReAct",
    )
with col4:
    v3_score = suite_metrics[v3_key]["overall_mean_score"]
    v1_score = suite_metrics[v1_key]["overall_mean_score"]
    st.metric(
        label="v3 Composite Score",
        value=f"{v3_score:.4f}",
        delta=f"+{v3_score - v1_score:.4f} vs v1 ReAct",
    )

st.markdown("---")

# Main Navigation Tabs
tab_trace, tab_metrics, tab_openrouter, tab_sandbox = st.tabs(
    [
        "🔍 Trajectory Trace Inspector",
        "📊 Benchmark Analytics & Charts",
        "⚡ OpenRouter Free AI Models",
        "🧪 Custom Trajectory Sandbox",
    ]
)

# TAB 1: Trace Inspector
with tab_trace:
    st.subheader("Interactive Multi-Step Trajectory Trace Inspector")

    # Filters
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        domains = ["All Domains"] + sorted(list({t.domain for t in dataset.tasks if hasattr(t, "domain")}))
        selected_domain = st.selectbox("Filter Domain", domains)

    filtered_tasks = dataset.tasks
    if selected_domain != "All Domains":
        filtered_tasks = [t for t in filtered_tasks if getattr(t, "domain", "") == selected_domain]

    with f_col2:
        task_options = {f"{t.task_id}: {t.goal[:65]}...": t for t in filtered_tasks}
        selected_task_label = st.selectbox("Select Task Scenario", list(task_options.keys()))
        current_task = task_options[selected_task_label]

    # Task Overview Card
    st.info(
        f"**Goal**: {current_task.goal}\n\n"
        f"**Domain**: `{getattr(current_task, 'domain', 'General')}` | "
        f"**Difficulty**: `{getattr(current_task, 'difficulty', 'Intermediate')}` | "
        f"**Optimal Steps**: `{current_task.optimal_steps}` | "
        f"**Reference Ground Truth**: *{current_task.reference_answer}*"
    )

    # Trajectory Comparison View
    st.markdown("### Execution Traces Across Agent Architectures")
    arch_tabs = st.tabs([
        "Agent v3 (Reflection & Recovery)",
        "Agent v2 (Structured Tools)",
        "Agent v1 (ReAct Baseline)",
        "⚡ Side-by-Side Comparison",
    ])

    def render_trajectory_steps(trajectory, final_answer, agent_name):
        if not trajectory:
            st.warning(f"No trajectory steps recorded for {agent_name}.")
            return

        for step in trajectory:
            status_val = step.status.value if isinstance(step.status, ToolCallStatus) else str(step.status)
            badge_class = "badge-success" if status_val == "Success" else "badge-danger"
            if "INVALID" in status_val.upper() or "WRONG" in status_val.upper():
                badge_class = "badge-warning"

            with st.expander(
                f"Step {step.step_number}: `{step.tool_name}` — [{status_val}]",
                expanded=True,
            ):
                st.markdown(f"**Thought:** *{step.thought}*")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Tool Arguments:**")
                    st.json(step.tool_args)
                with c2:
                    st.markdown(f"**Tool Output / Status: <span class='{badge_class}'>{status_val}</span>**", unsafe_allow_html=True)
                    st.code(step.tool_output, language="json" if "{" in step.tool_output else "text")

        st.markdown(f"**Final Answer Provided by Agent:** `{final_answer}`")

    with arch_tabs[0]:
        st.markdown("#### `Agent_v3_Reflection_Recovery`")
        render_trajectory_steps(current_task.trajectory_v3, current_task.final_answer_v3, "Agent v3")

    with arch_tabs[1]:
        st.markdown("#### `Agent_v2_Structured_Tools`")
        render_trajectory_steps(current_task.trajectory_v2, current_task.final_answer_v2, "Agent v2")

    with arch_tabs[2]:
        st.markdown("#### `Agent_v1_ReAct_Baseline`")
        render_trajectory_steps(current_task.trajectory_v1, current_task.final_answer_v1, "Agent v1")

    with arch_tabs[3]:
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            st.markdown("**v1 ReAct Baseline**")
            st.caption(f"Steps: {len(current_task.trajectory_v1)}")
            st.write(current_task.final_answer_v1)
        with s_col2:
            st.markdown("**v2 Structured Tools**")
            st.caption(f"Steps: {len(current_task.trajectory_v2)}")
            st.write(current_task.final_answer_v2)
        with s_col3:
            st.markdown("**v3 Reflection & Recovery**")
            st.caption(f"Steps: {len(current_task.trajectory_v3)}")
            st.write(current_task.final_answer_v3)

# TAB 2: Analytics & Charts
with tab_metrics:
    st.subheader("Controlled Empirical Benchmark Findings")

    # Table of suite metrics
    summary_rows = []
    for k, m in suite_metrics.items():
        summary_rows.append({
            "Architecture": k,
            "Success Rate (%)": f"{m['task_success_rate']*100:.1f}%",
            "Tool Accuracy (%)": f"{m['mean_tool_call_accuracy']*100:.1f}%",
            "Planning Efficiency (%)": f"{m['mean_planning_efficiency']*100:.1f}%",
            "Error Recovery (%)": f"{m['mean_error_recovery_rate']*100:.1f}%",
            "Composite Score": f"{m['overall_mean_score']:.4f}",
        })
    st.table(pd.DataFrame(summary_rows))

    # Figure Display
    st.markdown("### Publication Figures")
    fig_col1, fig_col2 = st.columns(2)

    fig_candidates = [
        ("reports/figures/agent_trajectory_performance.png", "Trajectory Success & Accuracy", fig_col1),
        ("reports/figures/planning_efficiency_breakdown.png", "Planning Efficiency & Recovery", fig_col2),
        ("reports/figures/tool_error_taxonomy.png", "Tool Error Outcome Distribution", fig_col1),
        ("reports/figures/domain_performance_comparison.png", "Enterprise Domain Breakdown", fig_col2),
    ]

    for rel_path, caption, col in fig_candidates:
        full_paths = [Path(rel_path), Path("repos/agentbench-works") / rel_path]
        found = False
        for fp in full_paths:
            if fp.exists():
                with col:
                    st.image(str(fp), caption=caption, use_container_width=True)
                found = True
                break

# TAB 3: OpenRouter Free Models
with tab_openrouter:
    st.subheader("⚡ Live OpenRouter Free AI Models Integration")
    st.markdown(
        "Run live agent trajectory evaluations with 100% free models on OpenRouter "
        "(`:free` tier, $0 cost, no credit card required)."
    )

    free_models = [
        "meta-llama/llama-3.3-70b-instruct:free",
        "deepseek/deepseek-r1:free",
        "deepseek/deepseek-chat:free",
        "qwen/qwen-2.5-72b-instruct:free",
        "google/gemini-2.0-flash-exp:free",
        "mistralai/mistral-7b-instruct:free",
    ]

    or_col1, or_col2 = st.columns([1, 2])
    with or_col1:
        chosen_model = st.selectbox("Select Free OpenRouter Model", free_models)
        env_key = os.getenv("OPENROUTER_API_KEY", "")
        masked_env = env_key[:8] + "..." if len(env_key) > 8 else "Not configured"
        st.write(f"**Env Key Status**: `{masked_env}`")

        user_api_key = st.text_input(
            "Or enter OpenRouter API Key for session",
            value="",
            type="password",
            help="Free keys can be created at https://openrouter.ai/keys",
        )
        active_key = user_api_key.strip() or env_key.strip()

    with or_col2:
        test_prompt = st.text_area(
            "Agent Task Prompt for Free Model",
            value="You are an autonomous cloud ops agent. Triage a crashlooping pod 'checkout-worker' in production cluster. Plan your multi-step tool calls in JSON.",
            height=120,
        )
        if st.button("🚀 Run Live Inference via OpenRouter", type="primary"):
            if not active_key:
                st.error("Please configure `OPENROUTER_API_KEY` in `.env` or paste it above to run live inference.")
            else:
                with st.spinner(f"Querying `{chosen_model}` via OpenRouter Free Tier..."):
                    try:
                        import requests
                        import time

                        headers = {
                            "Authorization": f"Bearer {active_key}",
                            "HTTP-Referer": "https://github.com/master72o",
                            "X-Title": "AgentBench Works",
                            "Content-Type": "application/json",
                        }
                        payload = {
                            "model": chosen_model,
                            "messages": [{"role": "user", "content": test_prompt}],
                            "temperature": 0.5,
                            "max_tokens": 800,
                        }
                        t0 = time.time()
                        resp = requests.post(
                            "https://openrouter.ai/api/v1/chat/completions",
                            headers=headers,
                            json=payload,
                            timeout=60,
                        )
                        latency = time.time() - t0

                        if resp.status_code == 200:
                            res_data = resp.json()
                            content = res_data["choices"][0]["message"]["content"]
                            st.success(f"Inference completed in {latency:.2f}s using `{chosen_model}`!")
                            st.markdown("### Generated Agent Output:")
                            st.write(content)
                        else:
                            st.error(f"OpenRouter API returned HTTP {resp.status_code}: {resp.text}")
                    except Exception as e:
                        st.error(f"Inference request failed: {e}")

# TAB 4: Custom Sandbox Evaluator
with tab_sandbox:
    st.subheader("🧪 Trajectory Evaluator Sandbox")
    st.markdown("Simulate and evaluate your own custom multi-step agent trajectories with the benchmark scoring engine.")

    sb_c1, sb_c2 = st.columns(2)
    with sb_c1:
        sb_goal = st.text_input("Task Goal", "Deploy security patch and verify health check.")
        sb_optimal = st.number_input("Optimal Steps", min_value=1, max_value=10, value=2)
        sb_ref = st.text_input("Ground Truth Reference Answer", "Security patch deployed and health check 200 OK.")
        sb_final = st.text_input("Agent Final Answer", "Security patch deployed and verified health 200 OK.")

    with sb_c2:
        st.markdown("**Trajectory Step Configuration**")
        n_steps = st.slider("Number of Steps Taken", 1, 6, 2)
        simulated_steps = []
        for i in range(1, n_steps + 1):
            s_c1, s_c2 = st.columns(2)
            with s_c1:
                t_name = st.text_input(f"Step {i} Tool", f"tool_action_{i}", key=f"t_{i}")
            with s_c2:
                t_stat = st.selectbox(
                    f"Step {i} Status",
                    ["Success", "Invalid_Arguments", "Tool_Execution_Error", "Wrong_Tool_Selected"],
                    key=f"s_{i}",
                )
            simulated_steps.append({
                "step_number": i,
                "thought": f"Executing step {i}",
                "tool_name": t_name,
                "tool_args": {"arg": 1},
                "tool_output": "output",
                "status": t_stat,
            })

    if st.button("Evaluate Custom Trajectory", type="primary"):
        from agent_eval.schema import TrajectoryStep, AgentTask
        task_mock = AgentTask(
            task_id="custom_01",
            goal=sb_goal,
            optimal_steps=int(sb_optimal),
            reference_answer=sb_ref,
        )
        steps_objs = [TrajectoryStep.from_dict(s) for s in simulated_steps]
        eval_res = TrajectoryEvaluator.evaluate_trajectory(
            task=task_mock,
            agent_version=AgentVariant.AGENT_V3_REFLECTION_RECOVERY,
            trajectory=steps_objs,
            final_answer=sb_final,
        )

        r1, r2, r3, r4, r5 = st.columns(5)
        r1.metric("Task Completed", "Yes" if eval_res.task_completed else "No")
        r2.metric("Tool Accuracy", f"{eval_res.tool_call_accuracy*100:.1f}%")
        r3.metric("Plan Efficiency", f"{eval_res.planning_efficiency*100:.1f}%")
        r4.metric("Error Recovery", f"{eval_res.error_recovery_rate*100:.1f}%")
        r5.metric("Composite Score", f"{eval_res.overall_score:.4f}")
