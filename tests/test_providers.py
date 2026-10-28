"""Provider selection + a full analysis over the OpenAI wire format (what Groq/OpenRouter use), HTTP mocked."""

import json
from pathlib import Path

try:  # openai>=3 ships its own HTTP client package, httpx2; older versions use httpx
    import httpx2 as httpx
except ImportError:
    import httpx
import pytest
from openai import AsyncOpenAI
from pydantic_ai.models.anthropic import AnthropicModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from data_analyst import analyze
from data_analyst.agent import make_model
from tests.test_agent import FINAL

DATA = str(Path(__file__).parents[1] / "data" / "sample_sales.csv")


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    for k in ("LLM_PROVIDER", "LLM_MODEL"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")


def test_default_is_anthropic():
    m = make_model()
    assert isinstance(m, AnthropicModel) and m.model_name == "claude-sonnet-5-5"


def test_groq_and_openrouter():
    g = make_model("groq")
    assert isinstance(g, OpenAIChatModel) and g.model_name == "openai/gpt-oss-120b"
    assert str(g.client.base_url).startswith("https://api.groq.com/openai/v1")
    o = make_model("openrouter", "qwen/qwen3-32b")
    assert o.model_name == "qwen/qwen3-32b" and "openrouter.ai" in str(o.client.base_url)


def test_errors(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY")
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        make_model("openrouter")
    with pytest.raises(ValueError, match="Unknown provider"):
        make_model("cohere")


def chat(tool_name, args, call_id):
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "openai/gpt-oss-120b",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {"name": tool_name, "arguments": json.dumps(args)},
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120},
    }


def test_full_analysis_over_openai_wire_format(tmp_path):
    requests = []

    def server(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            return httpx.Response(200, json=chat("profile_data", {}, "c1"))
        output_tool = next(t["function"]["name"] for t in body["tools"] if "final" in t["function"]["name"])
        return httpx.Response(200, json=chat(output_tool, {**FINAL, "charts": []}, "c2"))

    client = AsyncOpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key="gsk_test",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(server)),
    )
    model = OpenAIChatModel("openai/gpt-oss-120b", provider=OpenAIProvider(openai_client=client))
    report, ws = analyze(DATA, "Which region makes the most revenue?", str(tmp_path), model=model)

    assert report.confidence == "high" and ws.tool_calls == 1
    tool_names = {t["function"]["name"] for t in requests[0]["tools"]}
    assert {"profile_data", "clean_data", "aggregate", "make_chart"} <= tool_names
    tool_msg = next(m for m in requests[1]["messages"] if m["role"] == "tool")
    assert tool_msg["tool_call_id"] == "c1" and "Exact duplicate rows: 18" in tool_msg["content"]
