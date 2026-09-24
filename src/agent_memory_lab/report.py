"""Aggregation pipelines over the agent telemetry: cost and tokens per session, latency
percentiles per tool, and where a session's wall-clock time went. $percentile needs MongoDB 7.0+.

    uv run python -m agent_memory_lab.report
"""
from . import memory

P = [0.5, 0.95]


def per_session(d) -> list[dict]:
    return list(d.llm_calls.aggregate([
        {"$group": {"_id": "$session_id",
                    "llm_calls": {"$sum": 1},
                    "input_tokens": {"$sum": "$input_tokens"},
                    "output_tokens": {"$sum": "$output_tokens"},
                    "cache_read_tokens": {"$sum": "$cache_read_tokens"},
                    "cost_usd": {"$sum": "$cost_usd"},
                    "llm_ms": {"$sum": "$latency_ms"},
                    "first": {"$min": "$ts"}, "last": {"$max": "$ts"}}},
        # Join tool calls for the same session; the (session_id, ts) index serves the inner match.
        {"$lookup": {"from": "tool_calls", "localField": "_id", "foreignField": "session_id",
                     "pipeline": [{"$group": {"_id": None, "n": {"$sum": 1},
                                              "ms": {"$sum": "$latency_ms"},
                                              "errors": {"$sum": {"$cond": ["$ok", 0, 1]}}}}],
                     "as": "tools"}},
        {"$unwind": {"path": "$tools", "preserveNullAndEmptyArrays": True}},
        {"$sort": {"first": 1}},
        # $replaceWith (not $project) so the output fields come out in this order.
        {"$replaceWith": {"session_id": "$_id", "llm_calls": "$llm_calls",
                          "input_tokens": "$input_tokens", "output_tokens": "$output_tokens",
                          "cache_read_tokens": "$cache_read_tokens",
                          "cost_usd": {"$round": ["$cost_usd", 5]},
                          "tool_calls": {"$ifNull": ["$tools.n", 0]},
                          "tool_errors": {"$ifNull": ["$tools.errors", 0]},
                          "llm_ms": {"$round": ["$llm_ms", 0]},
                          "tool_ms": {"$round": [{"$ifNull": ["$tools.ms", 0]}, 0]}}},
    ]))


def per_tool(d) -> list[dict]:
    return list(d.tool_calls.aggregate([
        {"$group": {"_id": "$tool", "calls": {"$sum": 1},
                    "errors": {"$sum": {"$cond": ["$ok", 0, 1]}},
                    "pct": {"$percentile": {"input": "$latency_ms", "p": P, "method": "approximate"}},
                    "avg_result_chars": {"$avg": "$result_chars"}}},
        {"$sort": {"calls": -1}},
        {"$replaceWith": {"tool": "$_id", "calls": "$calls", "errors": "$errors",
                          "p50_ms": {"$round": [{"$arrayElemAt": ["$pct", 0]}, 1]},
                          "p95_ms": {"$round": [{"$arrayElemAt": ["$pct", 1]}, 1]},
                          "avg_result_chars": {"$round": ["$avg_result_chars", 0]}}},
    ]))


def llm_latency(d) -> list[dict]:
    return list(d.llm_calls.aggregate([
        {"$group": {"_id": "$model", "calls": {"$sum": 1},
                    "pct": {"$percentile": {"input": "$latency_ms", "p": P, "method": "approximate"}},
                    "out_tok_p50": {"$median": {"input": "$output_tokens", "method": "approximate"}}}},
        {"$replaceWith": {"model": "$_id", "calls": "$calls",
                          "p50_ms": {"$round": [{"$arrayElemAt": ["$pct", 0]}, 0]},
                          "p95_ms": {"$round": [{"$arrayElemAt": ["$pct", 1]}, 0]},
                          "out_tok_p50": "$out_tok_p50"}},
    ]))


def _table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    cols = list(rows[0])
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return "\n".join(lines)


def main() -> None:
    d = memory.db()
    for title, fn in [("Per session", per_session), ("Per tool", per_tool), ("LLM latency", llm_latency)]:
        print(f"\n### {title}\n\n{_table(fn(d))}")


if __name__ == "__main__":
    main()
