"""The autonomous analyst: a PydanticAI agent with typed tools and a typed result.

Give it a CSV and a question -> it profiles, cleans, analyzes, charts, and
returns an `AnalysisReport` (validated by Pydantic, so the report can't be
missing fields or hold the wrong types).
"""

from __future__ import annotations

import asyncio
import os
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelMessagesTypeAdapter
from pydantic_ai.models import Model
from pydantic_ai.usage import UsageLimits

from . import operations as ops
from .operations import ChartSpec, CleanStep, Metric, Workspace
from .trace import get_usage, run_with_trace

# provider -> (OpenAI-compatible base URL or None, env var with the key, default model)
PROVIDERS: dict[str, tuple[str | None, str, str]] = {
    "anthropic": (None, "ANTHROPIC_API_KEY", "claude-sonnet-5-5"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY", "openai/gpt-oss-120b"),
    # `openrouter/free` routes to a free model that supports tool calling
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY", "openrouter/free"),
}


def make_model(provider: str | None = None, model: str | None = None) -> Model:
    """Model for LLM_PROVIDER (default anthropic) and optional LLM_MODEL.

    The agent, its tools and its typed output don't change: only the model object does.
    """
    provider = (provider or os.getenv("LLM_PROVIDER", "anthropic")).lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider '{provider}'. Use one of: {', '.join(PROVIDERS)}")
    base_url, key_env, default_model = PROVIDERS[provider]
    model = model or os.getenv("LLM_MODEL") or default_model
    api_key = os.getenv(key_env)
    if not api_key or "your-key-here" in api_key:  # empty, or the placeholder copied from .env.example
        raise RuntimeError(
            f"Set a real {key_env} in .env to use the {provider} provider "
            "(or change LLM_PROVIDER to groq / openrouter / anthropic)"
        )

    if provider == "anthropic":
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.providers.anthropic import AnthropicProvider

        return AnthropicModel(model, provider=AnthropicProvider(api_key=api_key))

    # Groq and OpenRouter speak the OpenAI Chat Completions API
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(model, provider=OpenAIProvider(base_url=base_url, api_key=api_key))


MAX_TOOL_CALLS = 25


class Finding(BaseModel):
    statement: str = Field(description="One insight in plain language, with the number")
    evidence: str = Field(description="Which tool output supports it (e.g. 'aggregate revenue by region')")


class AnalysisReport(BaseModel):
    title: str
    answer: str = Field(description="Direct answer to the user's question in 1-3 sentences, with numbers")
    key_findings: list[Finding] = Field(min_length=1, max_length=6)
    data_quality_notes: list[str] = Field(description="What was wrong with the data and how it was fixed")
    charts: list[str] = Field(default_factory=list, description="Chart file names produced by make_chart")
    caveats: list[str] = Field(default_factory=list, description="Limitations, assumptions, what to check next")
    confidence: Literal["high", "medium", "low"]


INSTRUCTIONS = """You are a senior data analyst. You work ONLY through your tools - you cannot run code.
Workflow:
1. Call `profile_data` first. Look for wrong dtypes, currency stored as text, inconsistent
   categories (case/spaces), missing values, duplicates, mixed date formats.
2. Call `clean_data` with ALL fixes in one list. Each step is a JSON object with an "op" field (never a plain string). Derive the columns you need (e.g. revenue = quantity * unit_price).
3. Answer the question with `aggregate` / `time_series` / `correlation`. Every number you report must
   come from a tool output - never compute or estimate numbers yourself.
4. Make 1-3 charts that support the answer.
5. Return the AnalysisReport. Mention data quality problems and their impact in data_quality_notes.
Be concise and concrete. Prefer business language over statistics jargon."""


def build_agent(model=None) -> Agent[Workspace, AnalysisReport]:
    agent = Agent(
        model or make_model(),
        deps_type=Workspace,
        output_type=AnalysisReport,
        instructions=INSTRUCTIONS,
        retries=2,
    )

    def _guard(ctx: RunContext[Workspace]) -> None:
        ctx.deps.tool_calls += 1
        if ctx.deps.tool_calls > MAX_TOOL_CALLS:
            raise ModelRetry("Tool budget exhausted. Return the AnalysisReport now with what you have.")

    def _run(fn, *args, **kwargs) -> str:
        try:
            return fn(*args, **kwargs)
        except (KeyError, ValueError, TypeError) as e:
            # The model sees the error and retries with corrected arguments.
            raise ModelRetry(f"{type(e).__name__}: {e}") from e

    @agent.tool
    def profile_data(ctx: RunContext[Workspace]) -> str:
        """Schema, dtypes, null counts, unique counts, samples and duplicate count of the current data."""
        _guard(ctx)
        return ops.profile(ctx.deps)

    @agent.tool
    def clean_data(ctx: RunContext[Workspace], operations: list[CleanStep | str]) -> str:
        """Apply cleaning/derivation steps in order and return what each one changed.

        Each step is an OBJECT with an "op" field, e.g.
        [{"op": "drop_duplicates"}, {"op": "normalize_text", "column": "region", "case": "title"},
         {"op": "to_number", "column": "unit_price"}, {"op": "to_date", "column": "order_date", "dayfirst": true},
         {"op": "fill_missing", "column": "quantity", "strategy": "median"},
         {"op": "derive", "name": "revenue", "left": "quantity", "operator": "*", "right": "unit_price"}]
        """
        _guard(ctx)
        return _run(lambda: ops.apply_cleaning(ctx.deps, ops.to_clean_ops(operations)))

    @agent.tool
    def aggregate(
        ctx: RunContext[Workspace],
        group_by: list[str],
        metrics: list[Metric],
        sort_desc_by: str | None = None,
        top_n: int | None = None,
    ) -> str:
        """Group and aggregate. Output column names are '<column>_<agg>' (e.g. revenue_sum)."""
        _guard(ctx)
        return _run(ops.aggregate, ctx.deps, group_by, metrics, sort_desc_by, top_n)

    @agent.tool
    def time_series(
        ctx: RunContext[Workspace],
        date_column: str,
        value_column: str,
        freq: Literal["D", "W", "M", "Q"],
        agg: Literal["sum", "mean", "count"] = "sum",
    ) -> str:
        """Resample a value over time (D=day, W=week, M=month, Q=quarter). Date column must be cleaned first."""
        _guard(ctx)
        return _run(ops.time_series, ctx.deps, date_column, value_column, freq, agg)

    @agent.tool
    def correlation(ctx: RunContext[Workspace], columns: list[str]) -> str:
        """Pearson correlation matrix between numeric columns."""
        _guard(ctx)
        return _run(ops.correlation, ctx.deps, columns)

    @agent.tool
    def make_chart(ctx: RunContext[Workspace], spec: ChartSpec) -> str:
        """Save a chart (bar, line, hist, scatter) as PNG in the report folder."""
        _guard(ctx)
        return _run(ops.make_chart, ctx.deps, spec)

    @agent.output_validator
    def charts_must_exist(ctx: RunContext[Workspace], report: AnalysisReport) -> AnalysisReport:
        # Guardrail: the model can't cite a chart it never made.
        fake = [c for c in report.charts if c not in ctx.deps.charts]
        if fake:
            raise ModelRetry(f"These charts were never created: {fake}. Only cite {ctx.deps.charts}.")
        return report

    return agent


TOOL_NAMES = {"profile_data", "clean_data", "aggregate", "time_series", "correlation", "make_chart"}


def analyze(csv_path: str, question: str, output_dir: str = "output", model=None) -> tuple[AnalysisReport, Workspace]:
    ws = Workspace.from_csv(csv_path, output_dir)
    agent = build_agent(model)
    # agent.iter() instead of run_sync(): same loop, but we see every step (see trace.py)
    result, steps = asyncio.run(
        run_with_trace(
            agent,
            f"Dataset: {ws.source_name}\nQuestion: {question}",
            tool_names=TOOL_NAMES,
            deps=ws,
            usage_limits=UsageLimits(request_limit=30, total_tokens_limit=300_000),
        )
    )
    usage = get_usage(result)
    ws.trace_steps = steps
    ws.messages = ModelMessagesTypeAdapter.dump_python(result.all_messages(), mode="json")
    ws.usage = {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "requests": usage.requests}
    return result.output, ws
