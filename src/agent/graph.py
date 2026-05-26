"""LangGraph 그래프: 노드 연결 및 라우팅 로직.

라우팅 구조 (4가지):
  router ──┬── "learn"  → retrieval_agent → socratic_agent → tool_agent → composer → reviewer(조건부)
           ├── "chat"   → responder → END
           ├── "tools"  → tool_agent → composer → END  (reviewer 스킵)
           └── "escape" → pending_quiz 있으면 composer로 직행 (LLM 0회)
                          없으면 retrieval_agent → socratic_agent → composer → END

MAX_RETRIES: reviewer 실패 시 socratic_agent 재시도 최대 횟수
"""

import logging
from typing import Literal

from langgraph.graph import StateGraph, END

from src.agent.state import AgentState
from src.agent.core import router, composer, reviewer, should_review, responder
from src.agent.sub_agents import retrieval_agent, socratic_agent, tool_agent

logger = logging.getLogger("SocrAItes.Agent")

# reviewer 실패 시 socratic_agent 재시도 최대 횟수
MAX_RETRIES = 2

# ──────────────────────────────────────────────────────────────────────────────
# 라우팅 함수
# ──────────────────────────────────────────────────────────────────────────────

def route_after_router(
    state: AgentState,
) -> Literal["retrieval_agent", "responder", "tool_agent", "composer"]:
    """router 노드 실행 후 다음 노드를 결정한다.

    - "chat"   → responder (인사/잡담 즉시 응답)
    - "tools"  → tool_agent (명시적 도구 요청, retrieval/socratic 스킵)
    - "escape" + pending_quiz → composer (정답 직접 추출, LLM 0회)
    - "escape" + no quiz → retrieval_agent (정답 생성을 위해 socratic 경유)
    - "learn"  → retrieval_agent (기본 학습 경로 시작)
    """
    route = state.get("route", "learn")
    if route == "chat":
        logger.info("route=chat → responder로 분기.")
        return "responder"
    if route == "tools":
        logger.info("route=tools → tool_agent로 직접 분기 (retrieval/socratic 스킵).")
        return "tool_agent"
    if route == "escape":
        # 퀴즈 중 escape: pending_quiz 정답이 이미 있으므로 composer로 바로 이동
        if state.get("pending_quiz"):
            logger.info("route=escape + pending_quiz → composer로 직행 (LLM 0회).")
            return "composer"
        # 일반 escape: socratic_agent가 <answer>를 생성해야 하므로 retrieval부터 경유
        logger.info("route=escape (no quiz) → retrieval_agent → socratic_agent 경유.")
        return "retrieval_agent"
    # "learn" 또는 알 수 없는 값은 기본 학습 경로로 처리
    logger.info("route=learn → retrieval_agent로 분기.")
    return "retrieval_agent"


def route_after_composer(state: AgentState) -> Literal["reviewer", "end"]:
    """composer 노드 실행 후 reviewer 실행 여부를 결정한다.

    should_review(state) 헬퍼가 reviewer 실행 조건을 캡슐화한다:
    - tools/chat route → END (reviewer 항상 스킵)
    - learn route이고 (retry > 0 OR frustration >= 2 OR "tools" not in active_agents) → reviewer
    """
    if should_review(state):
        logger.info("reviewer 실행 조건 충족 → reviewer로 분기.")
        return "reviewer"
    logger.info("reviewer 스킵 → END.")
    return "end"


def route_after_reviewer(state: AgentState) -> Literal["socratic_agent", "end"]:
    """reviewer 노드 실행 후 재시도 여부를 결정한다.

    평가 실패(pass=False)이고 MAX_RETRIES 미만이면 socratic_agent로 돌아가 재생성.
    통과하거나 재시도 횟수를 초과했으면 END.
    """
    evaluation = state.get("evaluation", {})
    retry_count = state.get("retry_count", 0)
    if not evaluation.get("pass", True) and retry_count < MAX_RETRIES:
        logger.info(f"Reviewer 실패 — 재시도 #{retry_count} → socratic_agent.")
        return "socratic_agent"
    logger.info(f"Reviewer 통과 또는 최대 재시도 도달 (retry={retry_count}) → END.")
    return "end"


# ──────────────────────────────────────────────────────────────────────────────
# 그래프 빌드
# ──────────────────────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """모든 노드를 등록하고 엣지를 연결해 LangGraph StateGraph를 반환한다."""
    graph = StateGraph(AgentState)

    # ── 노드 등록 ─────────────────────────────────────────────────
    graph.add_node("router", router)                   # 쿼리 재작성 + 라우팅 (진입점)
    graph.add_node("retrieval_agent", retrieval_agent) # 벡터 스토어 검색
    graph.add_node("socratic_agent", socratic_agent)   # 소크라테스식 응답 생성
    graph.add_node("tool_agent", tool_agent)           # 학습 도구 호출
    graph.add_node("composer", composer)               # 서브에이전트 결과 조합
    graph.add_node("reviewer", reviewer)               # 품질 검사 (조건부)
    graph.add_node("responder", responder)             # 캐주얼 응답 (chat route)

    # ── 진입점 ───────────────────────────────────────────────────
    graph.set_entry_point("router")

    # ── router → 3방향 분기 ───────────────────────────────────────
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieval_agent": "retrieval_agent",  # learn / escape(no quiz) route
            "responder": "responder",              # chat route
            "tool_agent": "tool_agent",            # tools route
            "composer": "composer",                # escape + pending_quiz (fast path)
        },
    )

    # ── learn 경로: retrieval → socratic → tool → composer ───────
    graph.add_edge("retrieval_agent", "socratic_agent")
    graph.add_edge("socratic_agent", "tool_agent")

    # ── tool_agent → composer (tools/learn 경로 합류) ─────────────
    graph.add_edge("tool_agent", "composer")

    # ── composer → reviewer(조건부) 또는 END ─────────────────────
    graph.add_conditional_edges(
        "composer",
        route_after_composer,
        {
            "reviewer": "reviewer",
            "end": END,
        },
    )

    # ── reviewer → socratic_agent(재시도) 또는 END ───────────────
    graph.add_conditional_edges(
        "reviewer",
        route_after_reviewer,
        {
            "socratic_agent": "socratic_agent",
            "end": END,
        },
    )

    # ── chat route: responder → END ───────────────────────────────
    graph.add_edge("responder", END)

    return graph


# 모듈 로드 시 그래프를 빌드해 GRAPH 상수로 노출한다.
# api.py에서 GRAPH.compile()로 사용한다.
GRAPH = build_graph()
