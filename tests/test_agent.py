"""Agent tests with PydanticAI's FunctionModel: a scripted 'LLM' drives the real tools offline."""

from pathlib import Path

from pydantic_ai.messages import ModelResponse, RetryPromptPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from data_analyst import analyze
from data_analyst.report import render_markdown
from tests.test_operations import CLEANING

DATA = str(Path(__file__).parents[1] / "data" / "sample_sales.csv")

FINAL = {
    "title": "Revenue by region",
    "answer": "North America leads revenue.",
    "key_findings": [{"statement": "NA is #1", "evidence": "aggregate revenue by region"}],
    "data_quality_notes": ["18 duplicates removed"],
    "charts": ["rev.png"],
    "caveats": [],
    "confidence": "high",
}


def scripted(steps):
    """Each call returns the next scripted tool call; records what the model 'saw'."""
    seen = []

    def fn(messages, info: AgentInfo) -> ModelResponse:
        seen.append(messages)
        name, args = steps[len(seen) - 1]
        if name == "__final__":
            name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(name, args)])

    return FunctionModel(fn), seen


HAPPY_PATH = [
    ("profile_data", {}),
    ("clean_data", {"operations": CLEANING}),
    (
        "aggregate",
        {"group_by": ["region"], "metrics": [{"column": "revenue", "agg": "sum"}], "sort_desc_by": "revenue_sum"},
    ),
    ("make_chart", {"spec": {"kind": "bar", "x": "region", "y": "revenue", "title": "Revenue", "filename": "rev.png"}}),
    ("__final__", FINAL),
]


def tool_returns(messages):
    return [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, ToolReturnPart)]


def test_full_analysis_run(tmp_path):
    model, seen = scripted(HAPPY_PATH)
    report, ws = analyze(DATA, "Which region makes the most revenue?", str(tmp_path), model=model)
    assert report.confidence == "high"
    assert ws.tool_calls == 4
    assert "revenue" in ws.df.columns
    # The aggregate tool's real output reached the model before it answered
    agg_out = next(p.content for p in tool_returns(seen[-1]) if p.tool_name == "aggregate")
    assert "revenue_sum" in agg_out
    md = render_markdown(report, ws, "Which region makes the most revenue?").read_text()
    assert "![rev.png](rev.png)" in md and "drop_duplicates" in md


def test_bad_column_becomes_retry_prompt(tmp_path):
    steps = [
        ("aggregate", {"group_by": ["country"], "metrics": [{"column": "quantity", "agg": "sum"}]}),
        ("aggregate", {"group_by": ["product"], "metrics": [{"column": "quantity", "agg": "sum"}]}),
        ("__final__", {**FINAL, "charts": []}),
    ]
    model, seen = scripted(steps)
    analyze(DATA, "q", str(tmp_path), model=model)
    # The error goes back to the model as a retry/tool-error message, not a crash
    feedback = [
        str(p.content)
        for m in seen[1]
        for p in getattr(m, "parts", [])
        if isinstance(p, RetryPromptPart | ToolReturnPart)
    ]
    assert any("Unknown column" in f and "Available" in f for f in feedback)


def test_cannot_cite_charts_it_never_made(tmp_path):
    steps = [("__final__", FINAL), ("__final__", {**FINAL, "charts": []})]
    model, seen = scripted(steps)
    report, _ = analyze(DATA, "q", str(tmp_path), model=model)
    assert report.charts == []  # first answer rejected by the output validator
    assert len(seen) == 2
