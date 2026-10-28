"""The cleaning tool must be easy for small models: flat schema, strings tolerated, clear errors."""

import json
import logging
from pathlib import Path

import pytest

from data_analyst import analyze
from data_analyst.operations import CleanStep, to_clean_ops
from tests.test_agent import FINAL, scripted

DATA = str(Path(__file__).parents[1] / "data" / "sample_sales.csv")


def test_schema_seen_by_the_model_has_no_oneof():
    from pydantic import TypeAdapter

    schema = json.dumps(TypeAdapter(list[CleanStep | str]).json_schema())
    assert "oneOf" not in schema and "discriminator" not in schema


def test_shorthand_string_is_accepted():
    ops = to_clean_ops(["drop_duplicates", {"op": "to_number", "column": "unit_price"}])
    assert [o.op for o in ops] == ["drop_duplicates", "to_number"]


def test_string_that_needs_args_gives_fixable_error():
    with pytest.raises(ValueError, match="must be an object") as e:
        to_clean_ops(["to_number"])
    assert '"op": "drop_duplicates"' in str(e.value)  # shows an example


def test_missing_field_gives_fixable_error():
    with pytest.raises(ValueError, match=r"operations\[0\] \(to_number\): missing or invalid 'column'"):
        to_clean_ops([CleanStep(op="to_number")])


def test_agent_recovers_from_string_operations(tmp_path, caplog):
    """Exactly what gpt-oss did on Groq: strings instead of objects. Now it is a retry, not a crash."""
    steps = [
        ("clean_data", {"operations": ["normalize_text region", "to_number unit_price"]}),
        ("clean_data", {"operations": [{"op": "drop_duplicates"}, {"op": "to_number", "column": "unit_price"}]}),
        ("__final__", {**FINAL, "charts": []}),
    ]
    model, _ = scripted(steps)
    with caplog.at_level(logging.INFO, logger="analyst"):
        report, ws = analyze(DATA, "q", str(tmp_path), model=model)
    assert "OBSERVE ERROR clean_data -> " in caplog.text and "must be an object" in caplog.text
    assert ws.cleaning_log[0].startswith("drop_duplicates")
    assert report.confidence == "high"
