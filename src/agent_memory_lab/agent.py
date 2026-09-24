"""A small research agent over SciFact: LangGraph ReAct loop, Claude Haiku 4.5 on Bedrock,
conversation state checkpointed in MongoDB, long-term memory and telemetry in MongoDB, retrieval
from Qdrant (hybrid BM25 + dense). Needs the Qdrant index from `bench` and AWS credentials.

    uv run python -m agent_memory_lab.agent --session s1 "Does vitamin D reduce fracture risk?"
    uv run python -m agent_memory_lab.agent --demo      # scripted multi-session run
"""
import argparse
import uuid
import warnings

from langchain_aws import ChatBedrockConverse
from langchain_core.tools import tool
from langgraph.checkpoint.mongodb import MongoDBSaver
from langgraph.prebuilt import create_react_agent
from pymongo import MongoClient

from . import embed, memory, qdrant_store

# create_react_agent moved to langchain.agents.create_agent in LangGraph 1.0; it still works here
# and avoids pulling in the full langchain package.
warnings.filterwarnings("ignore", message="create_react_agent has been moved")

MODEL = "global.anthropic.claude-haiku-4-5-20251001-v1:0"
PROFILE, REGION = "claude-bedrock", "ap-south-1"

SYSTEM = """You answer questions about biomedical research claims using the SciFact abstracts.
Always call search_papers before answering a factual question and cite abstract ids like [12345].
If the user states a preference or a fact about themselves, call remember. Facts saved in earlier
conversations are listed below; call recall to search for older ones. Be concise."""


def build_tools(d, qc, user_id: str, session_id: str):
    @tool
    def search_papers(query: str) -> str:
        """Search 5,183 SciFact abstracts (hybrid BM25 + dense). Returns the top 5."""
        # No reranker: on SciFact the MS MARCO cross-encoder lowered nDCG@10 and added ~1.3 s (see bench).
        top = qdrant_store.hybrid(qc, embed.dense_query(query), embed.sparse_query(query), 5, with_text=True)
        return "\n\n".join(f"[{i}] {t}\n{b[:600]}" for i, t, b in top)

    @tool
    def remember(fact: str) -> str:
        """Save a durable fact or preference about the user for future conversations."""
        memory.remember(d, user_id, fact, session_id)
        return "saved"

    @tool
    def recall(query: str) -> str:
        """Look up facts previously saved about the user (keyword search)."""
        found = memory.recall(d, user_id, query)
        return "\n".join(f"- {m}" for m in found) or "nothing saved"

    return [search_papers, remember, recall]


def ask(question: str, session_id: str, user_id: str = "demo-user") -> str:
    client = MongoClient(memory.URI, tz_aware=True)
    d = client[memory.DB]
    memory.ensure_indexes(d)
    memory.start_session(d, session_id, user_id, MODEL)
    llm = ChatBedrockConverse(model=MODEL, credentials_profile_name=PROFILE, region_name=REGION,
                              temperature=0, max_tokens=1024)
    # Saved memories go into the system prompt on every turn rather than waiting for the model to
    # call recall: in the first demo run it never did, and answered without them.
    known = memory.recent(d, user_id)
    prompt = SYSTEM + "\n\nSaved facts about this user:\n" + ("\n".join(f"- {m}" for m in known) or "(none)")
    agent = create_react_agent(llm, build_tools(d, qdrant_store.client(), user_id, session_id),
                               prompt=prompt, checkpointer=MongoDBSaver(client, db_name=memory.DB))
    out = agent.invoke({"messages": [("user", question)]},
                       config={"configurable": {"thread_id": session_id},
                               "callbacks": [memory.MongoLogger(d, session_id, user_id, MODEL)]})
    return out["messages"][-1].content


# Three sessions for one user: the second turn depends on the checkpointed thread, the third
# session depends on long-term memory written in the first.
DEMO = [
    ("a", "I'm a pharmacist and I only care about human clinical evidence, not mouse studies. "
          "Does vitamin D supplementation reduce the risk of fractures?"),
    ("a", "What about in elderly people specifically?"),
    ("b", "Is there evidence that statins affect cancer risk?"),
    ("b", "Summarise that in two sentences."),
    ("c", "Do beta blockers help after a heart attack?"),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?")
    ap.add_argument("--session", default=None)
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()
    if args.demo:
        embed.dense_query("warm-up"), embed.sparse_query("warm-up")  # load ONNX models (~1.4 s) up front
        run = uuid.uuid4().hex[:6]
        for s, q in DEMO:
            print(f"\n[{run}-{s}] > {q}\n{ask(q, f'{run}-{s}')}")
    else:
        print(ask(args.question, args.session or uuid.uuid4().hex[:8]))


if __name__ == "__main__":
    main()
