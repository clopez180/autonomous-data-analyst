import json
import logging
import os
from pathlib import Path

import pytest

from data_analyst import analyze, cli
from tests.test_agent import HAPPY_PATH, scripted

DATA = str(Path(__file__).parents[1] / "data" / "sample_sales.csv")


def test_live_trace_shows_the_loop(tmp_path, caplog):
    model, _ = scripted(HAPPY_PATH)
    with caplog.at_level(logging.INFO, logger="analyst"):
        analyze(DATA, "Which region makes the most revenue?", str(tmp_path), model=model)
    log = caplog.text
    for expected in (
        "GOAL: Dataset: sample_sales.csv",
        "--- step 1",
        "ACT    profile_data({})",
        "OBSERVE profile_data -> Rows: 618",
        "ACT    clean_data(",
        "OBSERVE clean_data -> drop_duplicates",
        "ACT    aggregate(",
        "ACT    make_chart(",
        "ANSWER",
        "END: completed | 5 steps",
    ):
        assert expected in log, expected


def test_trace_steps_and_messages_are_kept(tmp_path):
    model, _ = scripted(HAPPY_PATH)
    _, ws = analyze(DATA, "q", str(tmp_path), model=model)
    model_steps = [s for s in ws.trace_steps if s["type"] == "model"]
    assert [s["calls"][0]["tool"] for s in model_steps[:4]] == ["profile_data", "clean_data", "aggregate", "make_chart"]
    observed = [s["tool"] for s in ws.trace_steps if s["type"] == "observe"]
    assert observed == ["profile_data", "clean_data", "aggregate", "make_chart"]
    assert ws.messages and ws.usage["requests"] == 5

    path = cli.save_trace({"steps": ws.trace_steps, "messages": ws.messages}, folder=tmp_path / "traces")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["steps"][0]["type"] == "model"


def test_errors_show_up_in_the_trace(tmp_path, caplog):
    from tests.test_agent import FINAL

    steps = [
        ("aggregate", {"group_by": ["country"], "metrics": [{"column": "quantity", "agg": "sum"}]}),
        ("__final__", {**FINAL, "charts": []}),
    ]
    model, _ = scripted(steps)
    with caplog.at_level(logging.INFO, logger="analyst"):
        analyze(DATA, "q", str(tmp_path), model=model)
    assert "OBSERVE ERROR aggregate -> KeyError: \"Unknown column(s) ['country']" in caplog.text


def test_load_dotenv(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("﻿LLM_PROVIDER=groq\nGROQ_API_KEY='gsk_x'\nOPENROUTER_API_KEY=\n# c\n", encoding="utf-8")
    for k in ("LLM_PROVIDER", "GROQ_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    cli.load_dotenv(env)
    assert os.environ["LLM_PROVIDER"] == "groq" and os.environ["GROQ_API_KEY"] == "gsk_x"
    assert "OPENROUTER_API_KEY" not in os.environ


def test_cli_exits_cleanly_without_key(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for k in ("LLM_PROVIDER", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr("sys.argv", ["analyst", DATA, "q"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "ANTHROPIC_API_KEY" in str(exc.value)


def test_placeholder_key_is_rejected(monkeypatch):
    from data_analyst.agent import make_model

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-your-key-here")
    with pytest.raises(RuntimeError, match="real ANTHROPIC_API_KEY"):
        make_model("anthropic")


def test_api_errors_become_short_messages():
    from pydantic_ai.exceptions import ModelHTTPError

    auth = ModelHTTPError(
        401,
        "claude-sonnet-5-5",
        {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}},
    )
    msg = cli.explain_error(auth)
    assert "HTTP 401" in msg and "invalid x-api-key" in msg and "LLM_PROVIDER" in msg

    bad_tool = ModelHTTPError(
        400, "openai/gpt-oss-120b", {"error": {"message": "Tool call validation failed", "code": "tool_use_failed"}}
    )
    assert "invalid tool call" in cli.explain_error(bad_tool)

    limit = ModelHTTPError(429, "openai/gpt-oss-120b", {"error": {"message": "Rate limit reached"}})
    assert "Wait a minute" in cli.explain_error(limit)


def test_cli_turns_api_failure_into_exit_message(tmp_path, monkeypatch):
    from pydantic_ai.exceptions import ModelHTTPError

    def boom(*a, **k):
        raise ModelHTTPError(401, "m", {"error": {"message": "invalid x-api-key"}})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cli, "make_model", lambda *a: object())
    monkeypatch.setattr(cli, "analyze", boom)
    monkeypatch.setattr("sys.argv", ["analyst", DATA, "q"])
    with pytest.raises(SystemExit) as exc:
        cli.main()
    assert "HTTP 401" in str(exc.value)
