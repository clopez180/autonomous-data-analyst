from pathlib import Path

import pandas as pd
import pytest

from data_analyst import operations as ops

DATA = Path(__file__).parents[1] / "data" / "sample_sales.csv"

CLEANING = [
    {"op": "drop_duplicates"},
    {"op": "normalize_text", "column": "region", "case": "title"},
    {"op": "to_number", "column": "unit_price"},
    {"op": "to_date", "column": "order_date", "dayfirst": True},
    {"op": "fill_missing", "column": "quantity", "strategy": "median"},
    {"op": "fill_missing", "column": "channel", "strategy": "value", "value": "unknown"},
    {"op": "derive", "name": "revenue", "left": "quantity", "operator": "*", "right": "unit_price"},
]


@pytest.fixture
def ws(tmp_path):
    return ops.Workspace.from_csv(DATA, tmp_path)


def clean(ws):
    from pydantic import TypeAdapter

    parsed = TypeAdapter(list[ops.CleanOp]).validate_python(CLEANING)
    return ops.apply_cleaning(ws, parsed)


def test_profile_reports_problems(ws):
    p = ops.profile(ws)
    assert "Exact duplicate rows: 18" in p
    assert "unit_price" in p and "nulls=" in p


def test_cleaning_pipeline_fixes_the_mess(ws):
    log = clean(ws)
    df = ws.df
    assert "removed 18 rows" in log
    assert set(df["region"].unique()) == {"Latam", "North America", "Europe"}
    assert pd.api.types.is_numeric_dtype(df["unit_price"]) and df["unit_price"].isna().sum() == 0
    assert pd.api.types.is_datetime64_any_dtype(df["order_date"]) and df["order_date"].isna().sum() == 0
    assert df["quantity"].isna().sum() == 0
    assert len(ws.cleaning_log) == len(CLEANING)


def test_aggregate_matches_pandas(ws):
    clean(ws)
    table = ops.aggregate(ws, ["region"], [ops.Metric(column="revenue", agg="sum")], "revenue_sum")
    expected = ws.df.groupby("region")["revenue"].sum().sort_values(ascending=False)
    first_line = table.splitlines()[2]
    assert expected.index[0] in first_line
    assert f"{expected.iloc[0]:.2f}" in first_line


def test_time_series_monthly(ws):
    clean(ws)
    out = ops.time_series(ws, "order_date", "revenue", "M", "sum")
    assert "2026-01-31" in out and "2026-09-30" in out


def test_unknown_column_is_a_clear_error(ws):
    with pytest.raises(KeyError, match="Available"):
        ops.aggregate(ws, ["country"], [ops.Metric(column="quantity", agg="sum")])


def test_chart_is_written(ws):
    clean(ws)
    msg = ops.make_chart(
        ws, ops.ChartSpec(kind="bar", x="region", y="revenue", title="Revenue by region", filename="rev.png")
    )
    assert (ws.output_dir / "rev.png").stat().st_size > 1000
    assert ws.charts == ["rev.png"] and "saved" in msg


def test_chart_filename_is_sanitized():
    with pytest.raises(ValueError):
        ops.ChartSpec(kind="bar", x="a", y="b", title="t", filename="../../etc/passwd")


def test_large_tables_are_truncated(ws):
    clean(ws)
    out = ops.aggregate(ws, ["order_id"], [ops.Metric(column="revenue", agg="sum")])
    assert "showing 25 of" in out
