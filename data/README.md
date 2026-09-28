# Autonomous Agent Trajectory Benchmark Dataset

This directory contains agent trajectory data capturing multi-step reasoning thoughts, tool calls, argument structures, tool execution outputs, and error recovery steps.

## Dataset Schema (`agent_trajectories.jsonl`)

```json
{
  "task_id": "ag_001",
  "goal": "Find weather in Tokyo and calculate flight cost from NYC.",
  "optimal_steps": 2,
  "reference_answer": "Tokyo weather is 22C and flight cost is $1200.",
  "trajectory_v1": [
    {
      "step_number": 1,
      "thought": "I need to get weather.",
      "tool_name": "get_weather",
      "tool_args": {"city": "Tokyo"},
      "tool_output": "Error: missing country code",
      "status": "Invalid_Arguments"
    }
  ],
  "final_answer_v1": "Could not retrieve weather.",
  "trajectory_v3": [
    {
      "step_number": 1,
      "thought": "I will search weather for Tokyo, JP.",
      "tool_name": "get_weather",
      "tool_args": {"city": "Tokyo", "country": "JP"},
      "tool_output": "Tokyo weather is 22C.",
      "status": "Success"
    },
    {
      "step_number": 2,
      "thought": "Now calculate flight cost NYC to Tokyo.",
      "tool_name": "get_flight_cost",
      "tool_args": {"origin": "NYC", "destination": "TYO"},
      "tool_output": "Flight cost is $1200.",
      "status": "Success"
    }
  ],
  "final_answer_v3": "Tokyo weather is 22C and flight cost is $1200."
}
```
