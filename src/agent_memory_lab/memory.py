"""MongoDB as the agent's operational store.

Conversation state (the message list) lives in LangGraph's MongoDBSaver checkpoints. Everything
around it lives in four collections of our own, one document per event:

    sessions    {_id: session_id, user_id, model, started_at, turns}
    llm_calls   {session_id, user_id, ts, model, input_tokens, output_tokens, cache_read_tokens,
                 latency_ms, cost_usd, stop_reason}
    tool_calls  {session_id, user_id, ts, tool, args, ok, error, latency_ms, result_chars}
    memories    {user_id, text, session_id, created_at}

One document per event (not an array pushed onto the session) keeps writes append-only, keeps
documents far below the 16 MB limit however long a session runs, and lets the aggregation
pipelines in report.py group by any field.
"""
import time
from datetime import datetime, timezone
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from pymongo import ASCENDING, DESCENDING, TEXT, MongoClient
from pymongo.database import Database

URI = "mongodb://localhost:27017"
DB = "agent_lab"

# USD per million tokens (Anthropic list prices). Unknown models are logged with cost_usd=None
# rather than a guessed number.
PRICES = {
    "global.anthropic.claude-haiku-4-5-20251001-v1:0": {"in": 1.00, "out": 5.00, "cache_read": 0.10},
}


def db(uri: str = URI, name: str = DB) -> Database:
    return MongoClient(uri, tz_aware=True)[name]


def ensure_indexes(d: Database) -> None:
    # Every report filters or groups by session first, then sorts by time.
    d.llm_calls.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
    d.tool_calls.create_index([("session_id", ASCENDING), ("ts", ASCENDING)])
    # Per-tool latency across all sessions ("is search_papers getting slower this week?").
    d.tool_calls.create_index([("tool", ASCENDING), ("ts", DESCENDING)])
    d.sessions.create_index([("user_id", ASCENDING), ("started_at", DESCENDING)])
    # recall() is a keyword search scoped to one user: equality prefix + text index.
    d.memories.create_index([("user_id", ASCENDING), ("text", TEXT)])
    d.memories.create_index([("user_id", ASCENDING), ("created_at", DESCENDING)])  # recent()


def now() -> datetime:
    return datetime.now(timezone.utc)


def cost(model: str, input_tokens: int, output_tokens: int, cache_read_tokens: int = 0) -> float | None:
    p = PRICES.get(model)
    if p is None:
        return None
    return round((input_tokens * p["in"] + output_tokens * p["out"] + cache_read_tokens * p["cache_read"]) / 1e6, 6)


def start_session(d: Database, session_id: str, user_id: str, model: str) -> None:
    d.sessions.update_one({"_id": session_id},
                          {"$setOnInsert": {"user_id": user_id, "model": model, "started_at": now()},
                           "$inc": {"turns": 1}}, upsert=True)


def remember(d: Database, user_id: str, text: str, session_id: str | None = None) -> None:
    d.memories.insert_one({"user_id": user_id, "text": text, "session_id": session_id, "created_at": now()})


def recent(d: Database, user_id: str, limit: int = 10) -> list[str]:
    cur = d.memories.find({"user_id": user_id}, {"text": 1}).sort("created_at", DESCENDING).limit(limit)
    return [m["text"] for m in cur]


def recall(d: Database, user_id: str, query: str, limit: int = 5) -> list[str]:
    cur = (d.memories.find({"user_id": user_id, "$text": {"$search": query}},
                           {"text": 1, "score": {"$meta": "textScore"}})
           .sort([("score", {"$meta": "textScore"})]).limit(limit))
    return [m["text"] for m in cur]


class MongoLogger(BaseCallbackHandler):
    """LangChain callback that writes one document per LLM call and per tool call."""

    def __init__(self, d: Database, session_id: str, user_id: str, model: str):
        self.d, self.session_id, self.user_id, self.model = d, session_id, user_id, model
        self._t0: dict[UUID, float] = {}
        self._tool: dict[UUID, tuple[str, str]] = {}

    def _base(self) -> dict:
        return {"session_id": self.session_id, "user_id": self.user_id, "ts": now()}

    def _elapsed(self, run_id: UUID) -> float:
        return round((time.perf_counter() - self._t0.pop(run_id, time.perf_counter())) * 1000, 1)

    def on_chat_model_start(self, serialized, messages, *, run_id, **kw):
        self._t0[run_id] = time.perf_counter()

    def on_llm_end(self, response, *, run_id, **kw):
        msg = response.generations[0][0].message
        u = getattr(msg, "usage_metadata", None) or {}
        cache_read = (u.get("input_token_details") or {}).get("cache_read", 0)
        # LangChain's input_tokens already includes cached tokens; bill them separately.
        fresh_in = u.get("input_tokens", 0) - cache_read
        self.d.llm_calls.insert_one({
            **self._base(), "model": self.model,
            "input_tokens": fresh_in, "output_tokens": u.get("output_tokens", 0),
            "cache_read_tokens": cache_read, "latency_ms": self._elapsed(run_id),
            "cost_usd": cost(self.model, fresh_in, u.get("output_tokens", 0), cache_read),
            "stop_reason": msg.response_metadata.get("stopReason"),
        })

    def on_tool_start(self, serialized, input_str, *, run_id, inputs=None, **kw):
        self._t0[run_id] = time.perf_counter()
        self._tool[run_id] = (serialized.get("name", "?"), inputs if inputs is not None else input_str)

    def _tool_doc(self, run_id, ok: bool, **extra) -> None:
        name, args = self._tool.pop(run_id, ("?", None))
        self.d.tool_calls.insert_one({**self._base(), "tool": name, "args": args, "ok": ok,
                                      "latency_ms": self._elapsed(run_id), **extra})

    def on_tool_end(self, output, *, run_id, **kw):
        content = getattr(output, "content", output)
        self._tool_doc(run_id, True, error=None, result_chars=len(str(content)))

    def on_tool_error(self, error, *, run_id, **kw):
        self._tool_doc(run_id, False, error=repr(error)[:500], result_chars=0)
