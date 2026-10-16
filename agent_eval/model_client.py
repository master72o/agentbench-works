"""
Multi-Model Client with Automatic Fallback for Agent Evaluation & Live Inference.
"""

import os
import json
import time
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import urllib.request
import urllib.error


DEFAULT_FREE_MODELS = [
    "google/gemini-2.0-flash-exp:free",
    "deepseek/deepseek-r1:free",
    "mistralai/mistral-7b-instruct:free",
    "qwen/qwen-2.5-72b-instruct:free",
]


@dataclass
class ModelCallResult:
    """Stores the execution result of an LLM query across models."""
    success: bool
    model_used: str
    content: str
    latency_ms: float
    raw_response: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    attempts: int = 1


class AgentModelClient:
    """
    Robust multi-model evaluation client with fallback chains.
    Queries OpenRouter or compatible endpoints, automatically retrying across models on failure.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        models: Optional[List[str]] = None,
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        timeout: float = 12.0,
    ):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.models = models or list(DEFAULT_FREE_MODELS)
        self.base_url = base_url
        self.timeout = timeout

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        mock: bool = False,
        mock_response: str = "I will search the database using tool: `query_database`.",
    ) -> ModelCallResult:
        """
        Executes a completion request across the fallback chain.
        If mock=True or api_key is missing without network, returns a deterministic mock response.
        """
        if mock or not self.api_key:
            return ModelCallResult(
                success=True,
                model_used=self.models[0] if self.models else "mock-agent-engine",
                content=mock_response,
                latency_ms=15.0,
                raw_response={"mock": True},
                attempts=1,
            )

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/master72o/agentbench-works",
            "X-Title": "AgentBench Works Trajectory Evaluator",
            "Content-Type": "application/json",
            "User-Agent": "AgentBench/1.1 (master72o)",
        }

        last_error = ""
        start_overall = time.perf_counter()

        for idx, model in enumerate(self.models, start=1):
            t0 = time.perf_counter()
            payload = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }

            req = urllib.request.Request(
                self.base_url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )

            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    latency = (time.perf_counter() - t0) * 1000
                    content = (
                        resp_data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return ModelCallResult(
                        success=True,
                        model_used=model,
                        content=content,
                        latency_ms=latency,
                        raw_response=resp_data,
                        attempts=idx,
                    )
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", errors="ignore")
                last_error = f"HTTP {e.code} on {model}: {body[:150]}"
                continue
            except Exception as e:
                last_error = f"Error on {model}: {str(e)}"
                continue

        total_latency = (time.perf_counter() - start_overall) * 1000
        return ModelCallResult(
            success=False,
            model_used="none",
            content="",
            latency_ms=total_latency,
            error=last_error or "All models in fallback chain failed",
            attempts=len(self.models),
        )

    @staticmethod
    def extract_tool_calls(text: str) -> List[Dict[str, Any]]:
        """
        Parses structured tool invocations from generated agent text.
        Supports JSON codeblocks, markdown fences, and embedded dictionary calls.
        """
        calls = []

        # 1. Look for ```json ... ``` blocks
        json_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        for block in json_blocks:
            try:
                data = json.loads(block)
                if isinstance(data, dict) and ("tool" in data or "action" in data or "name" in data):
                    calls.append(data)
            except Exception:
                pass

        # 2. Look for explicit `tool: name(args)` or tool inline patterns if no blocks found
        if not calls:
            inline = re.findall(r"(?:tool|call|action):\s*`?([a-zA-Z0-9_\-]+)`?", text, re.IGNORECASE)
            for tool_name in inline:
                calls.append({"tool": tool_name, "args": {}})

        return calls
