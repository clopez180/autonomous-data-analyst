"""`analyst data/sample_sales.csv "Which region drives revenue growth?" --trace`"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from pydantic_ai.exceptions import AgentRunError

from .agent import analyze, make_model
from .report import render_markdown


def load_dotenv(path: str | Path = ".env") -> None:
    """Load KEY=VALUE lines from .env into the environment (no extra dependency).

    Variables already set in the shell win, so you can still override per command.
    """
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and value:
            os.environ.setdefault(key, value)


def save_trace(data: dict, folder: str | Path = "traces") -> Path:
    """One JSON file per run: steps, tool results, tokens and the full conversation with the model."""
    out = Path(folder)
    out.mkdir(exist_ok=True)
    path = out / f"run-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')[:-3]}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return path


HINTS = {
    401: "The provider rejected your API key. Check LLM_PROVIDER and the matching key in .env.",
    403: "Your key has no access to this model. Try another model with --model.",
    404: "Model not found. Check the model id (--model / LLM_MODEL).",
    413: "Request too large for this model or tier. Try a model with a bigger context or a smaller CSV.",
    429: "Rate limit of the free tier. Wait a minute and run again, or use another model/provider.",
}


def explain_error(e: Exception) -> str:
    """Turn a model/API failure into a short message instead of a 60-line traceback."""
    status = getattr(e, "status_code", None)
    body = getattr(e, "body", None)
    detail = body.get("error", body) if isinstance(body, dict) else (body or str(e))
    if isinstance(detail, dict):
        detail = " | ".join(
            f"{k}: {v}" for k, v in detail.items() if k in ("type", "code", "message", "failed_generation")
        )
    lines = [f"Error: {type(e).__name__}" + (f" (HTTP {status})" if status else ""), f"  {str(detail)[:800]}"]
    hint = HINTS.get(status)
    if status == 400 and "tool" in str(detail).lower():
        hint = (
            "The model produced an invalid tool call (common with small/free models on complex schemas). "
            "Run again, or try another model, e.g. --model qwen/qwen3.8-27b, or --provider anthropic."
        )
    if hint:
        lines.append(f"  Hint: {hint}")
    return "\n".join(lines)


def main() -> None:
    p = argparse_parser()
    args = p.parse_args()

    load_dotenv()
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    trace_log = logging.getLogger("analyst")
    trace_log.addHandler(handler)
    trace_log.setLevel(logging.INFO if args.trace else logging.WARNING)
    trace_log.propagate = False

    try:
        model = make_model(args.provider, args.model)
    except (RuntimeError, ValueError) as e:  # missing key / unknown provider: short message, no traceback
        sys.exit(f"Error: {e}")

    started = datetime.now(UTC)
    try:
        report, ws = analyze(args.csv, args.question, args.out, model=model)
    except AgentRunError as e:
        sys.exit(explain_error(e))
    path = render_markdown(report, ws, args.question)

    if args.trace:
        print()
    print(report.answer)
    print(f"\nReport: {path}  |  charts: {', '.join(ws.charts) or 'none'}  |  tool calls: {ws.tool_calls}")
    print(f"-- tokens in/out: {ws.usage.get('input_tokens')}/{ws.usage.get('output_tokens')}")

    if not args.no_save:
        trace = save_trace(
            {
                "question": args.question,
                "csv": args.csv,
                "model": getattr(model, "model_name", str(model)),
                "started_at": started.isoformat(timespec="seconds"),
                "duration_s": round((datetime.now(UTC) - started).total_seconds(), 1),
                "usage": ws.usage,
                "steps": ws.trace_steps,
                "cleaning_log": ws.cleaning_log,
                "report": report.model_dump(),
                "messages": ws.messages,
            }
        )
        print(f"-- trace saved: {trace}")


def argparse_parser():
    import argparse

    p = argparse.ArgumentParser(description="Autonomous data analyst: CSV + question -> report.")
    p.add_argument("csv")
    p.add_argument("question")
    p.add_argument("--out", default="output")
    p.add_argument(
        "--provider",
        choices=["anthropic", "groq", "openrouter"],
        help="LLM provider (default: LLM_PROVIDER env var, else anthropic)",
    )
    p.add_argument("--model", help="Model id (default: LLM_MODEL env var, else the provider's default)")
    p.add_argument("--trace", action="store_true", help="Print every step live: think, act, observe")
    p.add_argument("--no-save", action="store_true", help="Don't save the run to traces/ (saved by default as JSON)")
    return p


if __name__ == "__main__":
    main()
