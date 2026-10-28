"""The analyst's toolbox: typed, deterministic pandas operations.

Design choice: the LLM never writes or executes Python. It composes these
validated operations instead. That removes the code-execution attack surface
(no sandbox needed), makes every step reproducible and logged, and keeps the
numbers in the report computed by pandas - not "estimated" by the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pydantic import BaseModel, Field, TypeAdapter, ValidationError

MAX_TABLE_ROWS = 25  # never dump a whole dataframe into the context window


@dataclass
class Workspace:
    """Mutable state shared by all tools during one analysis run (PydanticAI deps)."""

    df: pd.DataFrame
    output_dir: Path
    source_name: str = "dataset.csv"
    cleaning_log: list[str] = field(default_factory=list)
    charts: list[str] = field(default_factory=list)
    tool_calls: int = 0
    # filled after the run, for traces/*.json
    trace_steps: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)

    @classmethod
    def from_csv(cls, path: str | Path, output_dir: str | Path) -> Workspace:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        return cls(df=pd.read_csv(path), output_dir=out, source_name=Path(path).name)


def to_table(df: pd.DataFrame) -> str:
    shown = df.head(MAX_TABLE_ROWS)
    note = f"\n(showing {len(shown)} of {len(df)} rows)" if len(df) > MAX_TABLE_ROWS else ""
    return shown.to_markdown(index=False, floatfmt=".2f") + note


# ---------------------------------------------------------------------- profiling


def profile(ws: Workspace) -> str:
    df = ws.df
    cols = []
    for c in df.columns:
        s = df[c]
        sample = ", ".join(map(str, s.dropna().astype(str).unique()[:5]))
        cols.append(f"- {c}: dtype={s.dtype}, nulls={int(s.isna().sum())}, unique={s.nunique()}, sample=[{sample}]")
    dupes = int(df.duplicated().sum())
    return f"Rows: {len(df)} | Columns: {len(df.columns)} | Exact duplicate rows: {dupes}\n" + "\n".join(cols)


# ---------------------------------------------------------------------- cleaning


class DropDuplicates(BaseModel):
    op: Literal["drop_duplicates"] = "drop_duplicates"
    subset: list[str] | None = Field(default=None, description="Columns that identify a duplicate; null = all")


class DropMissing(BaseModel):
    op: Literal["drop_missing"] = "drop_missing"
    columns: list[str]


class FillMissing(BaseModel):
    op: Literal["fill_missing"] = "fill_missing"
    column: str
    strategy: Literal["median", "mean", "mode", "value"]
    value: str | float | None = None


class ToNumber(BaseModel):
    op: Literal["to_number"] = "to_number"
    column: str = Field(description="Strips currency symbols, thousands separators and spaces")


class ToDate(BaseModel):
    op: Literal["to_date"] = "to_date"
    column: str
    dayfirst: bool = Field(default=False, description="True for dd/mm/yyyy inputs")


class NormalizeText(BaseModel):
    op: Literal["normalize_text"] = "normalize_text"
    column: str
    case: Literal["title", "upper", "lower", "keep"] = "keep"


class DeriveColumn(BaseModel):
    op: Literal["derive"] = "derive"
    name: str
    left: str
    operator: Literal["*", "+", "-", "/"]
    right: str


CleanOp = Annotated[
    DropDuplicates | DropMissing | FillMissing | ToNumber | ToDate | NormalizeText | DeriveColumn,
    Field(discriminator="op"),
]


# ---------------------------------------------------------------------- the schema the model sees
#
# The strict union above is what the code uses. But a 7-way discriminated union (JSON-schema "oneOf")
# is hard for small/free models: they send strings instead of objects, and providers such as Groq
# reject the whole call. So the model gets ONE flat object with optional fields, and we convert it
# to the strict types ourselves. Any mistake becomes a clear error message the model can fix.

CleanOpName = Literal[
    "drop_duplicates", "drop_missing", "fill_missing", "to_number", "to_date", "normalize_text", "derive"
]


class CleanStep(BaseModel):
    """One cleaning step. Only `op` is required; fill the fields that op needs."""

    op: CleanOpName = Field(
        description=(
            "drop_duplicates [subset] | drop_missing [columns] | fill_missing [column, strategy, value] | "
            "to_number [column] | to_date [column, dayfirst] | normalize_text [column, case] | "
            "derive [name, left, operator, right]"
        )
    )
    column: str | None = None
    columns: list[str] | None = Field(default=None, description="drop_missing: columns to check")
    subset: list[str] | None = Field(default=None, description="drop_duplicates: key columns, omit for all")
    strategy: Literal["median", "mean", "mode", "value"] | None = Field(default=None, description="fill_missing")
    value: str | float | None = Field(default=None, description="fill_missing with strategy=value")
    dayfirst: bool | None = Field(default=None, description="to_date: true for dd/mm/yyyy")
    case: Literal["title", "upper", "lower", "keep"] | None = Field(default=None, description="normalize_text")
    name: str | None = Field(default=None, description="derive: new column name")
    left: str | None = Field(default=None, description="derive: left column")
    operator: Literal["*", "+", "-", "/"] | None = Field(default=None, description="derive")
    right: str | None = Field(default=None, description="derive: right column")


CLEAN_EXAMPLE = (
    '[{"op": "drop_duplicates"}, {"op": "to_number", "column": "unit_price"}, '
    '{"op": "derive", "name": "revenue", "left": "quantity", "operator": "*", "right": "unit_price"}]'
)
_NO_ARGS = {"drop_duplicates"}
_STRICT = TypeAdapter(list[CleanOp])


def to_clean_ops(steps: list[CleanStep | str | dict]) -> list[CleanOp]:
    """Flat steps from the model -> strict typed operations. Raises ValueError with a fixable message."""
    raw: list[dict] = []
    for i, step in enumerate(steps):
        if isinstance(step, str):  # shorthand like "drop_duplicates" from weaker models
            name = step.strip().lower()
            if name in _NO_ARGS:
                raw.append({"op": name})
                continue
            raise ValueError(
                f"operations[{i}] is the string {step!r}; each operation must be an object like "
                f"{{'op': 'to_number', 'column': 'unit_price'}}. Example list: {CLEAN_EXAMPLE}"
            )
        data = step.model_dump(exclude_none=True) if isinstance(step, CleanStep) else dict(step)
        raw.append(data)
    try:
        return _STRICT.validate_python(raw)
    except ValidationError as e:
        problems = "; ".join(
            f"operations[{err['loc'][0]}] ({raw[err['loc'][0]].get('op')}): missing or invalid '{err['loc'][-1]}'"
            if len(err["loc"]) > 1 and isinstance(err["loc"][0], int)
            else err["msg"]
            for err in e.errors()
        )
        raise ValueError(f"Invalid cleaning steps: {problems}. Example: {CLEAN_EXAMPLE}") from None


def _require(df: pd.DataFrame, *cols: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Unknown column(s) {missing}. Available: {list(df.columns)}")


def apply_cleaning(ws: Workspace, ops: list[CleanOp]) -> str:
    df = ws.df.copy()
    messages = []
    for op in ops:
        before = len(df)
        if isinstance(op, DropDuplicates):
            if op.subset:
                _require(df, *op.subset)
            df = df.drop_duplicates(subset=op.subset)
            msg = f"drop_duplicates({op.subset or 'all columns'}): removed {before - len(df)} rows"
        elif isinstance(op, DropMissing):
            _require(df, *op.columns)
            df = df.dropna(subset=op.columns)
            msg = f"drop_missing({op.columns}): removed {before - len(df)} rows"
        elif isinstance(op, FillMissing):
            _require(df, op.column)
            n = int(df[op.column].isna().sum())
            col = df[op.column]
            if op.strategy == "value":
                fill = op.value
            elif op.strategy == "mode":
                fill = col.mode().iloc[0]
            else:
                fill = getattr(col, op.strategy)()
            df[op.column] = df[op.column].fillna(fill)
            msg = f"fill_missing({op.column}, {op.strategy}={fill}): filled {n} values"
        elif isinstance(op, ToNumber):
            _require(df, op.column)
            cleaned = df[op.column].astype(str).str.replace(r"[^\d.\-]", "", regex=True)
            df[op.column] = pd.to_numeric(cleaned.replace("", None), errors="coerce")
            msg = f"to_number({op.column}): {int(df[op.column].isna().sum())} values unparseable -> NaN"
        elif isinstance(op, ToDate):
            _require(df, op.column)
            raw = df[op.column].astype(str)
            iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
            other = pd.to_datetime(raw, dayfirst=op.dayfirst, errors="coerce", format="mixed")
            df[op.column] = iso.fillna(other)
            msg = f"to_date({op.column}): {int(df[op.column].isna().sum())} values unparseable -> NaT"
        elif isinstance(op, NormalizeText):
            _require(df, op.column)
            s = df[op.column].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            if op.case != "keep":
                s = getattr(s.str, op.case)()
            df[op.column] = s
            msg = f"normalize_text({op.column}, {op.case}): {df[op.column].nunique()} distinct values now"
        elif isinstance(op, DeriveColumn):
            _require(df, op.left, op.right)
            a, b = df[op.left], df[op.right]
            df[op.name] = {"*": a * b, "+": a + b, "-": a - b, "/": a / b}[op.operator]
            msg = f"derive {op.name} = {op.left} {op.operator} {op.right}"
        messages.append(msg)
    ws.df = df
    ws.cleaning_log += messages
    return "\n".join(messages) + f"\nRows now: {len(df)}"


# ---------------------------------------------------------------------- analysis


class Metric(BaseModel):
    column: str
    agg: Literal["sum", "mean", "median", "count", "min", "max", "nunique"]


def aggregate(
    ws: Workspace,
    group_by: list[str],
    metrics: list[Metric],
    sort_desc_by: str | None = None,
    top_n: int | None = None,
) -> str:
    df = ws.df
    _require(df, *group_by, *(m.column for m in metrics))
    named = {f"{m.column}_{m.agg}": pd.NamedAgg(column=m.column, aggfunc=m.agg) for m in metrics}
    out = (
        df.groupby(group_by, dropna=False).agg(**named).reset_index()
        if group_by
        else df.agg({m.column: m.agg for m in metrics}).to_frame("value").reset_index()
    )
    if sort_desc_by and sort_desc_by in out.columns:
        out = out.sort_values(sort_desc_by, ascending=False)
    if top_n:
        out = out.head(top_n)
    return to_table(out)


def time_series(ws: Workspace, date_column: str, value_column: str, freq: Literal["D", "W", "M", "Q"], agg: str) -> str:
    _require(ws.df, date_column, value_column)
    s = ws.df.set_index(pd.to_datetime(ws.df[date_column]))[value_column]
    period = {"D": "D", "W": "W", "M": "ME", "Q": "QE"}[freq]
    out = s.resample(period).agg(agg).reset_index()
    out[date_column] = out[date_column].dt.date
    return to_table(out)


def correlation(ws: Workspace, columns: list[str]) -> str:
    _require(ws.df, *columns)
    return ws.df[columns].corr(numeric_only=True).round(3).to_markdown()


# ---------------------------------------------------------------------- charts


class ChartSpec(BaseModel):
    kind: Literal["bar", "line", "hist", "scatter"]
    x: str
    y: str | None = None
    agg: Literal["sum", "mean", "count", "median"] | None = Field(
        default="sum", description="For bar/line: aggregate y by x"
    )
    time_bucket: Literal["D", "W", "M", "Q"] | None = Field(
        default=None, description="If x is a date column: bucket by day/week/month/quarter before plotting"
    )
    title: str
    filename: str = Field(pattern=r"^[a-z0-9_\-]+\.png$", description="snake_case file name ending in .png")


def make_chart(ws: Workspace, spec: ChartSpec) -> str:
    df = ws.df
    _require(df, spec.x, *([spec.y] if spec.y else []))
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    if spec.kind == "hist":
        df[spec.x].dropna().plot.hist(ax=ax, bins=20)
    elif spec.kind == "scatter":
        df.plot.scatter(x=spec.x, y=spec.y, ax=ax, alpha=0.6)
    else:
        if spec.time_bucket:
            period = {"D": "D", "W": "W", "M": "ME", "Q": "QE"}[spec.time_bucket]
            data = df.set_index(pd.to_datetime(df[spec.x]))[spec.y].resample(period).agg(spec.agg or "sum")
            data.index = data.index.strftime("%Y-%m-%d" if spec.time_bucket in "DW" else "%b %Y")
        else:
            data = df.groupby(spec.x)[spec.y].agg(spec.agg or "sum")
        if spec.time_bucket and spec.kind == "line":
            data.plot.line(ax=ax, marker="o")
        elif spec.kind == "bar":
            data.sort_values(ascending=False).plot.bar(ax=ax)
        else:
            data.sort_index().plot.line(ax=ax, marker="o")
    ax.set_title(spec.title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = ws.output_dir / spec.filename
    fig.savefig(path)
    plt.close(fig)
    ws.charts.append(spec.filename)
    return f"Chart saved: {spec.filename}"
