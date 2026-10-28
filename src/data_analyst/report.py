"""Render the typed AnalysisReport to a Markdown file next to its charts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from .agent import AnalysisReport
from .operations import Workspace


def render_markdown(report: AnalysisReport, ws: Workspace, question: str) -> Path:
    lines = [
        f"# {report.title}",
        "",
        (
            f"_Source: `{ws.source_name}` · Generated {datetime.now(UTC).date().isoformat()} by autonomous-data-analyst · "
            f"Confidence: **{report.confidence}**_"
        ),
        "",
        f"> **Question:** {question}",
        "",
        "## Answer",
        report.answer,
        "",
        "## Key findings",
        *[f"{i}. {f.statement}  \n   <sub>Evidence: {f.evidence}</sub>" for i, f in enumerate(report.key_findings, 1)],
        "",
    ]
    for chart in report.charts:
        lines += [f"![{chart}]({chart})", ""]
    lines += ["## Data quality", *[f"- {n}" for n in report.data_quality_notes], ""]
    lines += ["<details><summary>Cleaning log (exact operations applied)</summary>", ""]
    lines += [f"- `{step}`" for step in ws.cleaning_log] + ["", "</details>", ""]
    if report.caveats:
        lines += ["## Caveats & next steps", *[f"- {c}" for c in report.caveats], ""]
    path = ws.output_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
