"""Runs against the docker-compose MongoDB in a throwaway database; skipped if it's not up."""
import uuid
from types import SimpleNamespace

import pytest
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError

from agent_memory_lab import memory, report

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"


@pytest.fixture
def d():
    client = MongoClient(memory.URI, serverSelectionTimeoutMS=1000, tz_aware=True)
    try:
        client.admin.command("ping")
    except ServerSelectionTimeoutError:
        pytest.skip("MongoDB not running (docker compose up -d)")
    name = f"test_{uuid.uuid4().hex[:8]}"
    db = client[name]
    memory.ensure_indexes(db)
    yield db
    client.drop_database(name)


def test_cost_uses_list_prices():
    # 1M in + 1M out on Haiku 4.5 = $1 + $5.
    assert memory.cost(MODEL, 1_000_000, 1_000_000) == 6.0
    assert memory.cost("unknown-model", 10, 10) is None


def test_recall_is_scoped_to_user(d):
    memory.remember(d, "u1", "prefers human clinical trials over mouse studies")
    memory.remember(d, "u2", "prefers mouse studies")
    assert memory.recall(d, "u1", "mouse studies") == ["prefers human clinical trials over mouse studies"]
    assert memory.recall(d, "u1", "cardiology") == []


def _llm_result(inp, out, cache=0):
    msg = SimpleNamespace(usage_metadata={"input_tokens": inp, "output_tokens": out,
                                          "input_token_details": {"cache_read": cache}},
                          response_metadata={"stopReason": "end_turn"})
    return SimpleNamespace(generations=[[SimpleNamespace(message=msg)]])


def test_logger_and_reports(d):
    for sid in ("s1", "s2"):
        memory.start_session(d, sid, "u1", MODEL)
        log = memory.MongoLogger(d, sid, "u1", MODEL)
        for _ in range(2 if sid == "s1" else 1):
            r = uuid.uuid4()
            log.on_chat_model_start({}, [], run_id=r)
            log.on_llm_end(_llm_result(1200, 300, cache=200), run_id=r)
        t = uuid.uuid4()
        log.on_tool_start({"name": "search_papers"}, "q", run_id=t, inputs={"query": "q"})
        log.on_tool_end(SimpleNamespace(content="x" * 50), run_id=t)
    t = uuid.uuid4()
    log.on_tool_start({"name": "recall"}, "q", run_id=t)
    log.on_tool_error(RuntimeError("boom"), run_id=t)

    call = d.llm_calls.find_one()
    assert call["input_tokens"] == 1000 and call["cache_read_tokens"] == 200  # cached split out

    sessions = {s["session_id"]: s for s in report.per_session(d)}
    assert sessions["s1"]["llm_calls"] == 2 and sessions["s1"]["tool_calls"] == 1
    assert sessions["s2"]["tool_errors"] == 1
    assert sessions["s1"]["cost_usd"] == pytest.approx(2 * memory.cost(MODEL, 1000, 300, 200))

    tools = {t["tool"]: t for t in report.per_tool(d)}
    assert tools["search_papers"]["calls"] == 2 and tools["search_papers"]["avg_result_chars"] == 50
    assert tools["recall"]["errors"] == 1
    assert report.llm_latency(d)[0]["calls"] == 3
    assert d.sessions.find_one({"_id": "s1"})["turns"] == 1


def test_recent_is_newest_first_and_scoped(d):
    for text in ("first", "second", "third"):
        memory.remember(d, "u1", text)
    memory.remember(d, "u2", "other user")
    assert memory.recent(d, "u1", limit=2) == ["third", "second"]
