import io
import json
import urllib.error
from unittest.mock import patch, MagicMock
from agent_eval.model_client import AgentModelClient, ModelCallResult
from agent_eval.report_generator import ReportGenerator
from agent_eval.schema import AgentVariant, TrajectoryEvaluation


def test_mock_generation():
    client = AgentModelClient(api_key="test-key")
    res = client.generate([{"role": "user", "content": "hello"}], mock=True)
    assert res.success is True
    assert "mock" in res.content.lower() or "search" in res.content.lower()
    assert res.latency_ms > 0
    assert res.attempts == 1


def test_tool_call_extraction_json():
    text = """
Here is my plan:
```json
{
  "tool": "calculator",
  "args": {"expression": "2 + 2"}
}
```
Done!
"""
    tools = AgentModelClient.extract_tool_calls(text)
    assert len(tools) == 1
    assert tools[0]["tool"] == "calculator"
    assert tools[0]["args"]["expression"] == "2 + 2"


def test_tool_call_extraction_inline():
    text = "Executing tool: search_engine to find customer record."
    tools = AgentModelClient.extract_tool_calls(text)
    assert len(tools) == 1
    assert tools[0]["tool"] == "search_engine"


def test_fallback_chain_on_error():
    models = ["model-primary-failing", "model-backup-succeeding"]
    client = AgentModelClient(api_key="sk-real-dummy", models=models)

    # First call fails with 429, second succeeds
    fail_resp = urllib.error.HTTPError(
        url="http://test",
        code=429,
        msg="Rate limit",
        hdrs={},
        fp=io.BytesIO(b'{"error": "rate limited"}'),
    )

    success_resp_data = {
        "choices": [{"message": {"content": "Final recovery plan"}}]
    }
    success_mock = MagicMock()
    success_mock.read.return_value = json.dumps(success_resp_data).encode("utf-8")
    success_mock.__enter__.return_value = success_mock

    with patch("urllib.request.urlopen", side_effect=[fail_resp, success_mock]):
        res = client.generate([{"role": "user", "content": "Execute step"}])
        assert res.success is True
        assert res.attempts == 2
        assert res.model_used == "model-backup-succeeding"
        assert res.content == "Final recovery plan"


def test_export_summary_dict():
    eval1 = TrajectoryEvaluation(
        task_id="task_001",
        agent_version=AgentVariant.AGENT_V3_REFLECTION_RECOVERY,
        task_completed=True,
        tool_call_accuracy=1.0,
        planning_efficiency=0.9,
        argument_correctness=1.0,
        error_recovery_rate=1.0,
        overall_score=0.95,
        steps_taken=3,
        optimal_steps=3,
    )
    all_evals = {AgentVariant.AGENT_V3_REFLECTION_RECOVERY: [eval1]}
    summary = ReportGenerator.export_summary_dict(all_evals)
    key = AgentVariant.AGENT_V3_REFLECTION_RECOVERY.value
    assert key in summary
    assert summary[key]["total_tasks"] == 1
    assert summary[key]["task_success_rate"] == 1.0
    assert summary[key]["mean_overall_score"] == 0.95
