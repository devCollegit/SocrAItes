"""Sub Agent — Retrieval: queries Elasticsearch vector store."""
import logging
from typing import List, Dict, Any

from src.agent.state import AgentState
from src.agent.helpers import _log_trace
from src.rag import vectorstore

logger = logging.getLogger("SocrAItes.Agent")


def retrieval_agent(state: AgentState) -> AgentState:
    """Retrieves relevant lecture chunks when 'retrieval' is in sub_agents."""
    logger.info("--- [Retrieval Agent] Step ---")
    if "retrieval" not in state.get("sub_agents", ["retrieval"]):
        state["retrieved_docs"] = []
        _log_trace(step="RetrievalAgent", purpose="Skipped.", decision="Not in sub_agents.")
        return state

    query = state.get("contextualized_query", "") or (
        state["messages"][-1]["content"] if state.get("messages") else ""
    )
    results = vectorstore.query(query, k=5)
    state["retrieved_docs"] = results

    logger.info(f"Retrieved {len(results)} chunks.")
    _log_trace(
        step="RetrievalAgent",
        purpose="Query Elasticsearch and return relevant lecture chunks.",
        inputs={"Query": query},
        decision={"Count": len(results), "Snippets": [r["text"][:80] + "..." for r in results]},
    )
    return state
