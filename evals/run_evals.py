"""Evals for the autonomous analyst.

Two kinds of checks per case:
1. Correctness (deterministic): a reference solution computed with plain pandas
   says what the right answer is; we check the agent's answer names it and
   quotes the number within 1%.
2. Faithfulness (LLM-as-a-judge): a separate judge model reads the tool outputs the
   agent actually saw and checks every finding is supported by them.

Usage: python evals/run_evals.py   (needs ANTHROPIC_API_KEY)
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.usage import UsageLimits

sys.path.insert(0, str(Path(__file__).parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).parents[1]))

from data_analyst import Workspace, build_agent  # noqa: E402
from data_analyst.agent import make_model  # noqa: E402
from tests.test_operations import CLEANING  # noqa: E402  (reference cleaning)

HERE = Path(__file__).parent
DATA = HERE.parent / "data" / "sample_sales.csv"


def reference_answer(ref: dict) -> tuple[str, float]:
    from pydantic import TypeAdapter

    from data_analyst.operations import CleanOp, apply_cleaning

    with tempfile.TemporaryDirectory() as tmp:
        ws = Workspace.from_csv(DATA, tmp)
        apply_cleaning(ws, TypeAdapter(list[CleanOp]).validate_python(CLEANING))
        df = ws.df
    if ref["kind"] == "top_group":
        s = df.groupby(ref["group_by"])[ref["value"]].agg(ref["agg"]).sort_values(ascending=False)
        return str(s.index[0]), float(s.iloc[0])
    s = df.set_index(df[ref["date"]])[ref["value"]].resample("ME").sum().sort_values(ascending=False)
    return s.index[0].strftime("%B"), float(s.iloc[0])


def mentions_number(text: str, expected: float, tol: float = 0.01) -> bool:
    for raw in re.findall(r"\d[\d,]*\.?\d*", text):
        try:
            val = float(raw.replace(",", ""))
        except ValueError:
            continue
        if abs(val - expected) <= abs(expected) * tol:
            return True
    return False


class Verdict(BaseModel):
    faithful: bool
    unsupported_claims: list[str]


judge = Agent(
    make_model(),  # same provider as the analyst (LLM_PROVIDER); use a stronger one for judging if you can
    output_type=Verdict,
    instructions=(
        "You audit a data analysis. Given the raw tool outputs and the analyst's findings, "
        "mark faithful=false if ANY number or claim in the findings is not supported by the tool outputs."
    ),
)


def main() -> int:
    cases = json.loads((HERE / "cases.json").read_text())
    agent = build_agent()
    results = []
    for case in cases:
        expected_label, expected_value = reference_answer(case["reference"])
        with tempfile.TemporaryDirectory() as tmp:
            ws = Workspace.from_csv(DATA, tmp)
            run = agent.run_sync(case["question"], deps=ws, usage_limits=UsageLimits(request_limit=30))
        report = run.output
        tool_outputs = [
            f"[{p.tool_name}]\n{p.content}"
            for m in run.all_messages()
            for p in getattr(m, "parts", [])
            if isinstance(p, ToolReturnPart)
        ]
        findings = "\n".join(f"- {f.statement}" for f in report.key_findings) + f"\nAnswer: {report.answer}"
        verdict = judge.run_sync(f"TOOL OUTPUTS:\n{chr(10).join(tool_outputs)}\n\nFINDINGS:\n{findings}").output
        usage = run.usage()
        results.append(
            {
                "id": case["id"],
                "correct_label": expected_label.lower() in report.answer.lower(),
                "correct_number": mentions_number(report.answer, expected_value),
                "faithful": verdict.faithful,
                "unsupported": verdict.unsupported_claims,
                "tool_calls": ws.tool_calls,
                "tokens": usage.total_tokens,
            }
        )
        print(json.dumps(results[-1], ensure_ascii=False))

    score = sum(r["correct_label"] and r["correct_number"] and r["faithful"] for r in results) / len(results)
    print(f"\nScore: {score:.0%} ({len(results)} cases)")
    return 0 if score >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
