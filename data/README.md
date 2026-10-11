# Autonomous Multi-Step Agent Trajectory Benchmark Dataset Card

## Dataset Summary

`data/agent_trajectories.jsonl` contains **16 enterprise-grade multi-step agent benchmark tasks** evaluating autonomous tool calling, parameter schema adherence, planning efficiency, and runtime error recovery.

Each task records multi-step execution traces across three agent architectures:
1. **`Agent_v1_ReAct_Baseline`**: Unstructured Reason + Act prompting.
2. **`Agent_v2_Structured_Tools`**: ReAct equipped with strict Pydantic JSON Schema tool argument validation.
3. **`Agent_v3_Reflection_Recovery`**: Structured tool calling paired with a Self-Reflection & Replanning error recovery loop.

---

## Domain Coverage

The 16 benchmark scenarios span 5 critical enterprise domains:

| Domain | Task IDs | Core Challenges Tested |
| :--- | :--- | :--- |
| **Finance & Treasury** | `ag_001`, `ag_002`, `ag_003` | FX rate conversions, VAT reverse charge, compound interest formulas, travel expense policy audits. |
| **Cloud DevOps** | `ag_004`, `ag_005`, `ag_006` | Kubernetes pod crashloop triage (OOMKilled), weighted Route53 DNS failover, CDN cache purges, AWS IAM wildcard privilege audits. |
| **Database & Systems Reliability** | `ag_007`, `ag_008`, `ag_009` | Dead-letter queue (DLQ) replay, Redlock distributed lock contention, ElasticSearch unassigned shard rebalancing. |
| **Cybersecurity & Threat Response** | `ag_010`, `ag_011`, `ag_012` | Leaked API key revocation, session JWT invalidation, WAF CIDR block rules, CVE dependency semver patching. |
| **Enterprise Operations & CRM** | `ag_013`, `ag_014`, `ag_015`, `ag_016` | Missing proof-of-delivery credit memos, multi-timezone meeting scheduling, ERP inventory reorder points, foreign transit receipt OCR. |

---

## Failure Taxonomy

The benchmark explicitly evaluates 6 distinct autonomous agent failure modes:

- `Argument Hallucination`: Passing unprompted or hallucinated argument keys.
- `Schema Type Mismatch`: Passing string values where floats/ints or nested dicts are required.
- `Syntax Error in Expression`: Using non-standard operators (e.g., `^` instead of `**`).
- `Namespace / Context Scope Error`: Searching for resources in default namespaces instead of target environments.
- `Runtime Resource Lock Contention`: Handling transient concurrency locks (Redlock) or rate-limit timeouts.
- `Downstream Cascading Error`: Blindly continuing with corrupt intermediate state rather than triggering recovery.

---

## Schema Definition

Each line in `data/agent_trajectories.jsonl` follows this JSON structure:

```json
{
  "task_id": "ag_001",
  "domain": "Finance & Treasury",
  "difficulty": "Intermediate",
  "failure_types_tested": ["Argument Hallucination", "Tool Syntax Error"],
  "goal": "Reconcile German invoice #INV-4091: convert EUR 14,500 to USD at 1.085 FX rate, apply 19% VAT reverse charge, and format ledger entry.",
  "optimal_steps": 3,
  "reference_answer": "Reconciled INV-4091: USD amount is $15,732.50, reverse-charge VAT is $2,989.18, total balanced ledger record posted.",
  "trajectory_v1": [
    {
      "step_number": 1,
      "thought": "Query invoice details.",
      "tool_name": "get_invoice",
      "tool_args": {"id": "INV-4091"},
      "tool_output": "{\"invoice_id\": \"INV-4091\", ...}",
      "status": "Success"
    }
  ],
  "final_answer_v1": "Unable to complete currency conversion due to parameter errors.",
  "trajectory_v2": [...],
  "final_answer_v2": "...",
  "trajectory_v3": [...],
  "final_answer_v3": "..."
}
```

---

## Benchmark Reproducibility

To re-run the benchmark and generate updated results:

```bash
python -m agent_eval.cli run \
  --dataset data/agent_trajectories.jsonl \
  --output-dir results \
  --report-path reports/agent_evaluation_report.md \
  --figures-dir reports/figures
```

License: MIT
