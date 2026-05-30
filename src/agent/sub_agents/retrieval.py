"""Sub Agent — Retrieval: Elasticsearch 벡터 스토어에서 강의 자료를 검색.

active_agents에 "retrieval"이 포함된 경우에만 실행한다.
검색 쿼리는 router가 생성한 rewritten_query를 사용한다.
"""

import logging

from src.agent.state import AgentState
from src.agent.helpers import _log_trace
from src.rag import vectorstore

logger = logging.getLogger("SocrAItes.Agent")


def retrieval_agent(state: AgentState) -> AgentState:
    """벡터 스토어에서 관련 강의 청크를 검색한다. 'retrieval'이 active_agents에 없으면 스킵."""
    logger.info("--- [Retrieval Agent] Step ---")

    # active_agents에 "retrieval"이 없으면 이 턴은 검색 불필요
    if "retrieval" not in state.get("active_agents", ["retrieval"]):
        state["retrieved_docs"] = []
        _log_trace(
            step="RetrievalAgent",
            purpose="스킵.",
            decision="active_agents에 'retrieval' 없음.",
        )
        return state

    # 선택된 문서가 없으면 검색을 건너뜀
    selected_docs = state.get("selected_docs")
    if selected_docs is not None and len(selected_docs) == 0:
        state["retrieved_docs"] = []
        _log_trace(
            step="RetrievalAgent",
            purpose="스킵 (체크된 강의 자료 없음).",
            decision="selected_docs가 비어있음.",
        )
        return state

    # router가 재작성한 쿼리 사용, 없으면 원본 메시지 폴백
    query = state.get("rewritten_query", "") or (
        state["messages"][-1]["content"] if state.get("messages") else ""
    )
    results = vectorstore.query(query, k=5, selected_sources=selected_docs)
    state["retrieved_docs"] = results

    logger.info(f"검색 완료: {len(results)}개 청크.")
    _log_trace(
        step="RetrievalAgent",
        purpose="Elasticsearch 쿼리로 관련 강의 청크 반환.",
        inputs={"Query": query},
        decision={
            "Count": len(results),
            "Snippets": [r["text"][:80] + "..." for r in results],
        },
    )
    return state
