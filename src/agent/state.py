"""Agent state: LangGraph 노드 간 전달되는 공유 상태 정의.

이 모듈은 LangGraph 노드들이 주고받는 불변 스냅샷(AgentState)을 정의한다.
TypedDict를 사용하므로 JSON 직렬화와 정적 타입 체크가 동시에 가능하다.
total=False 설정으로 모든 필드는 선택적(optional)이며, DEFAULT_STATE로 초기화한다.
"""

from __future__ import annotations

from typing import TypedDict, List, Dict, Any


class AgentState(TypedDict, total=False):
    """LangGraph 노드 간 공유 상태 TypedDict.

    ── 대화 기본 정보 ──────────────────────────────────────────
    messages        : 전체 대화 히스토리 (role/content dict 리스트)
    session_id      : SQLite DB 세션 UUID
    user_profile    : 사용자 학습 프로필 (학습 스타일, 어조, 배경 등)

    ── Router 출력 ─────────────────────────────────────────────
    route           : "learn" | "chat" | "tools"  — 이번 턴 라우팅 결정
    rewritten_query : 검색 최적화용으로 재작성된 쿼리 (한국어)
    active_agents   : 이번 턴 실행할 서브에이전트 목록 ["retrieval","socratic","tools"]
    subtask         : 이번 턴 수행할 작업 설명 (한국어)

    ── 학습 상태 추적 ───────────────────────────────────────────
    socratic_depth      : 소크라테스 깊이 (0=Light, 1=Standard, 2=Deep)
    frustration_level   : 누적 좌절 신호 수 (높을수록 힌트 강화)
    retry_count         : reviewer 재시도 횟수 (MAX_RETRIES 초과 시 종료)

    ── Sub-agent 출력 ───────────────────────────────────────────
    retrieved_docs  : retrieval_agent 검색 결과 (벡터 스토어 청크 리스트)
    tutor_response  : socratic_agent 소크라테스 응답 문자열
    tool_result     : tool_agent 도구 호출 결과 {analysis, tool_results, tool_calls_made}

    ── 최종 출력 ────────────────────────────────────────────────
    response        : composer가 조립한 최종 응답 문자열 (API가 클라이언트로 전송)
    evaluation      : reviewer 품질 평가 결과 {pass, feedback, scores}

    ── 기타 ─────────────────────────────────────────────────────
    tool_results    : 도구 실행 결과 목록 (SSE 스트림 API 응답용)
    pending_quiz    : 채점 대기 중인 퀴즈 아이템 리스트 (세션 간 복원 필요)
    """

    # ── 대화 기본 정보 ──────────────────────────────────────────
    messages: List[Dict[str, Any]]
    session_id: str
    user_profile: Dict[str, Any]

    # ── Router 출력 ─────────────────────────────────────────────
    route: str
    rewritten_query: str
    active_agents: List[str]
    subtask: str

    # ── 학습 상태 추적 ───────────────────────────────────────────
    socratic_depth: int
    frustration_level: int
    retry_count: int

    # ── Sub-agent 출력 ───────────────────────────────────────────
    retrieved_docs: List[Any]
    tutor_response: str
    tool_result: Dict[str, Any]

    # ── 최종 출력 ────────────────────────────────────────────────
    response: str
    evaluation: Dict[str, Any]

    # ── 기타 ─────────────────────────────────────────────────────
    tool_results: List[Dict[str, Any]]
    pending_quiz: List[Dict[str, Any]]
    force_explain: bool
    selected_docs: List[str]


# 그래프 실행 전 초기 상태값.
# DEFAULT_STATE.copy() 후 필요한 필드만 override해서 사용한다.
DEFAULT_STATE: AgentState = {
    "messages": [],
    "session_id": "",
    "user_profile": {},
    "route": "",
    "rewritten_query": "",
    "active_agents": [],
    "subtask": "",
    "socratic_depth": 1,
    "frustration_level": 0,
    "retry_count": 0,
    "retrieved_docs": [],
    "tutor_response": "",
    "tool_result": {},
    "response": "",
    "evaluation": {},
    "tool_results": [],
    "pending_quiz": [],
    "force_explain": False,
    "selected_docs": [],
}
