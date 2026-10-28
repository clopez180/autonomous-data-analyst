"""Watch PydanticAI's agent loop step by step.

`agent.run_sync()` hides the loop. `agent.iter()` exposes it: each node is one
phase of the same ReAct loop you wrote by hand in agent-from-scratch.

    ModelRequestNode  -> we send the model: the question, or tool results (OBSERVE)
    CallToolsNode     -> the model answered: thoughts (THINK) + tool calls (ACT)
    End               -> the model called the output tool: the typed AnalysisReport

Every step is logged live (logger "analyst", shown with --trace) and also
collected as plain dicts so the CLI can save them to traces/*.json.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import RetryPromptPart, TextPart, ThinkingPart, ToolCallPart, ToolReturnPart

log = logging.getLogger("analyst")


def _short(text: Any, limit: int = 300) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[:limit] + f"... (+{len(text) - limit} chars)"


def get_usage(obj: Any) -> Any:
    """`usage` is a method in some PydanticAI versions and a property in others."""
    usage = obj.usage
    return usage() if callable(usage) else usage


def _args(part: ToolCallPart) -> Any:
    args = part.args
    if isinstance(args, str):
        try:
            return json.loads(args)
        except json.JSONDecodeError:
            return args
    return args


async def run_with_trace(agent: Agent, prompt: str, *, tool_names: set[str], **run_kwargs) -> tuple[Any, list[dict]]:
    """Run the agent node by node. Returns (run result, steps)."""
    steps: list[dict] = []
    step_n = 0
    t_node = time.perf_counter()
    log.info("GOAL: %s", _short(prompt, 500))

    async with agent.iter(prompt, **run_kwargs) as run:
        async for node in run:
            if Agent.is_model_request_node(node):
                # What goes back to the model: results of the tools it asked for (OBSERVE)
                for part in node.request.parts:
                    failed = isinstance(part, RetryPromptPart) or (
                        isinstance(part, ToolReturnPart) and getattr(part, "outcome", "success") != "success"
                    )
                    if isinstance(part, ToolReturnPart | RetryPromptPart) and (failed or part.tool_name in tool_names):
                        # A failed tool call goes back to the model as feedback so it can fix its arguments
                        label = "OBSERVE ERROR" if failed else "OBSERVE"
                        log.info("  %s %s -> %s", label, part.tool_name or "output", _short(part.content))
                        steps.append(
                            {
                                "step": step_n,
                                "type": "error" if failed else "observe",
                                "tool": part.tool_name,
                                "content": str(part.content),
                            }
                        )
                t_node = time.perf_counter()

            elif Agent.is_call_tools_node(node):
                # The model's answer for this step: thoughts and tool calls
                step_n += 1
                resp = node.model_response
                usage = resp.usage
                model_ms = int((time.perf_counter() - t_node) * 1000)
                log.info(
                    "--- step %d | model %.1fs | tokens %d -> %d",
                    step_n,
                    model_ms / 1000,
                    usage.input_tokens or 0,
                    usage.output_tokens or 0,
                )
                record = {
                    "step": step_n,
                    "type": "model",
                    "model_ms": model_ms,
                    "input_tokens": usage.input_tokens or 0,
                    "output_tokens": usage.output_tokens or 0,
                    "think": [],
                    "calls": [],
                }
                for part in resp.parts:
                    if isinstance(part, ThinkingPart | TextPart) and part.content.strip():
                        log.info("  THINK  %s", _short(part.content))
                        record["think"].append(part.content)
                    elif isinstance(part, ToolCallPart):
                        args = _args(part)
                        if part.tool_name in tool_names:
                            log.info("  ACT    %s(%s)", part.tool_name, _short(json.dumps(args, ensure_ascii=False)))
                        else:
                            log.info("  ANSWER %s -> typed AnalysisReport (validated by Pydantic)", part.tool_name)
                        record["calls"].append({"tool": part.tool_name, "args": args})
                steps.append(record)

            elif Agent.is_end_node(node):
                usage = get_usage(run)
                log.info(
                    "END: completed | %d steps | tokens in/out %d/%d",
                    step_n,
                    usage.input_tokens or 0,
                    usage.output_tokens or 0,
                )

    return run.result, steps
