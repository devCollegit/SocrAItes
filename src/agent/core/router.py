"""Core Agent — Router: 쿼리 재작성 + 라우팅 + 에이전트 선택을 단일 LLM 호출로 처리.

기존 Coordinator + Planner 두 노드를 하나로 병합하여 LLM 호출 횟수를 절반으로 줄인다.

출력:
- route          : "learn" | "chat" | "tools" | "escape"
- rewritten_query: 검색 최적화된 한국어 재작성 쿼리
- active_agents  : 이번 턴 실행할 서브에이전트 목록
- subtask        : 이번 턴 수행할 작업 설명 (한국어)
"""

import os
import re
import json
import logging
from typing import List

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import (
    _log_trace,
    _detect_frustration,
    _is_last_message_quiz_prompt,
    _format_history,
)

logger = logging.getLogger("SocrAItes.Agent")

# ──────────────────────────────────────────────────────────────────────────────
# 상수 및 키워드 목록
# ──────────────────────────────────────────────────────────────────────────────

# rule-based fallback용: 이 키워드가 포함되면 "tools" route로 강제 분기
_TOOL_KEYWORDS = [
    "퀴즈", "문제 내", "문제내", "quiz", "연습문제",  # 퀴즈 생성
    "약점", "약점 저장", "weakness",                  # 약점 저장
    "복습 일정", "일정 등록", "schedule",              # 일정 등록
]

# ──────────────────────────────────────────────────────────────────────────────
# LLM 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

ROUTER_PROMPT = """You are the Router for SocrAItes, a Socratic learning assistant.

Analyze the conversation and latest user message, then output a single JSON with:
1. rewritten_query : 최신 메시지를 검색 최적화된 독립적인 한국어 쿼리로 재작성.
   - 이전 맥락을 반영해 단독으로 이해할 수 있어야 함.
   - 인사/잡담이면 원문 그대로 반환.
   - "다시", "또", "계속" 등의 반복 요청이면 다음 순서 개념을 검색하도록 재작성.

2. route : 아래 우선순위 순서대로 하나를 선택.

   ⚠️ CRITICAL — 가장 먼저 판단 (다른 맥락보다 우선):
   - "escape" : 학생이 소크라테스 방식을 명시적으로 거부하고 직접 답을 요구할 때.
     퀴즈 중이든, 학습 중이든, 어떤 맥락이든 상관없이 escape 의도가 감지되면 반드시 "escape".
     예시 표현 (이에 국한되지 않음):
       "정답 알려줘", "답 알려줘", "그냥 알려줘", "답을 알고 싶어",
       "힌트 말고", "소크라테스 말고", "직접 알려줘", "모르겠으니까 그냥 알려줘",
       "설명 말고 답만", "답을 알려주세요"

   나머지 (escape가 아닌 경우):
   - "tools"  : 퀴즈 생성, 약점/강점 저장, 복습 일정 등록 등 명시적 도구 요청.
   - "chat"   : 인사, 감사, 잡담 등 학습과 무관한 캐주얼 대화.
   - "learn"  : 강의 자료 기반 소크라테스식 학습 (기본값).

3. active_agents : route=="learn"일 때만 의미 있음.
   - 기본: ["retrieval", "socratic"]
   - 반복 오개념 또는 도구 필요 시: ["retrieval", "socratic", "tools"]
   route가 "learn" 이외이면 반드시 [].

4. subtask : 이번 턴 수행할 작업 설명 (한국어 1~2문장).

Frustration level: {frustration_level} (0=none, 높을수록 더 좌절함)
Socratic depth: {depth_mode}
Evaluator feedback (retry 시): "{eval_feedback}"

Conversation History:
{history_text}

Latest User Message:
{last_message}

Respond ONLY in JSON:
{{
  "rewritten_query": "<한국어 쿼리>",
  "route": "learn" | "chat" | "tools" | "escape",
  "active_agents": ["retrieval", "socratic"],
  "subtask": "<한국어 작업 설명>"
}}"""

# ──────────────────────────────────────────────────────────────────────────────
# 내부 유틸리티
# ──────────────────────────────────────────────────────────────────────────────

def _needs_tools(text: str) -> bool:
    """텍스트에 도구 키워드가 포함되어 있으면 True 반환 (rule-based fallback용)."""
    clean = text.replace(" ", "").lower()
    return any(k.replace(" ", "").lower() in clean for k in _TOOL_KEYWORDS)


def _parse_router_json(content: str) -> dict:
    """LLM 응답 문자열에서 JSON을 파싱한다. 마크다운 코드 펜스를 먼저 제거."""
    # ```json ... ``` 또는 ``` ... ``` 형태의 코드 펜스 제거
    cleaned = re.sub(r"^```(?:json)?\s*", "", content.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    return json.loads(cleaned)


# ──────────────────────────────────────────────────────────────────────────────
# 메인 노드 함수
# ──────────────────────────────────────────────────────────────────────────────

def router(state: AgentState) -> AgentState:
    """쿼리 재작성 + 라우팅 + 에이전트 선택을 단일 LLM 호출로 수행.

    처리 순서:
    1. 좌절 감지 — 퀴즈 중 좌절 시 LLM 없이 즉시 "learn" 라우팅
    2. LLM 호출 — ROUTER_PROMPT로 JSON 응답 생성
    3. JSON 파싱 — 실패 시 rule-based fallback 적용
    4. state 업데이트 및 반환
    """
    logger.info("--- [Router] Step ---")
    messages = state.get("messages", [])
    last_message = messages[-1]["content"] if messages else ""

    # ── 1. 좌절 레벨 계산 ────────────────────────────────────────
    # 전체 대화에서 사용자의 좌절 신호 횟수를 누적한다.
    frustration_count = sum(
        1 for msg in messages
        if msg.get("role") == "user" and _detect_frustration(msg.get("content", ""))
    )
    state["frustration_level"] = frustration_count

    # ── 2. 퀴즈 중 좌절 시 즉시 learn 라우팅 ────────────────────
    # LLM 호출 없이 바로 learn route + active_agents 기본값 적용.
    pending_quiz = state.get("pending_quiz", [])
    is_quiz_pending = len(pending_quiz) > 0 or _is_last_message_quiz_prompt(messages[:-1])
    is_frustrated_in_quiz = (
        is_quiz_pending
        and messages
        and messages[-1].get("role") == "user"
        and _detect_frustration(messages[-1].get("content", ""))
    )

    if is_frustrated_in_quiz:
        # 퀴즈 중 좌절: 소크라테스 힌트 제공을 위해 learn route로 강제 분기
        state["rewritten_query"] = last_message
        state["route"] = "learn"
        state["active_agents"] = ["retrieval", "socratic"]
        state["subtask"] = "퀴즈 중 좌절 감지 — 힌트와 격려를 제공하며 소크라테스식 유도"
        logger.info("퀴즈 중 좌절 감지 — LLM 없이 learn으로 즉시 라우팅.")
        _log_trace(
            step="Router",
            purpose="퀴즈 중 좌절 감지로 LLM 바이패스",
            inputs={"Message": last_message, "Frustration": frustration_count},
            decision="route=learn (LLM 없이 즉시 결정)",
        )
        return state

    # ── 3. LLM 호출 ─────────────────────────────────────────────
    depth_modes = ["Light (1-2 turns)", "Standard (3-4 turns)", "Deep (5+ turns)"]
    depth_mode = depth_modes[state.get("socratic_depth", 1)]
    eval_feedback = state.get("evaluation", {}).get("feedback", "")
    is_first_turn = len(messages) <= 1
    history_text = _format_history(messages)

    prompt = ROUTER_PROMPT.format(
        frustration_level=frustration_count,
        depth_mode=depth_mode,
        eval_feedback=eval_feedback,
        history_text=history_text,
        last_message=last_message,
    )

    # OpenAI API 키가 있으면 JSON 모드 사용, 없으면 일반 호출 (FakeLLM 등)
    if os.getenv("OPENAI_API_KEY"):
        response = llm.bind(response_format={"type": "json_object"}).invoke(
            [HumanMessage(content=prompt)]
        )
    else:
        response = llm.invoke([HumanMessage(content=prompt)])

    response_str = _get_content(response).strip()

    # ── 4. JSON 파싱 + rule-based fallback ──────────────────────
    try:
        parsed = _parse_router_json(response_str)
        rewritten_query = parsed.get("rewritten_query", "").strip()
        route = parsed.get("route", "learn").strip().lower()
        active_agents = parsed.get("active_agents", [])
        subtask = parsed.get("subtask", "")
    except Exception as e:
        # 파싱 실패 시 원문 사용 + 키워드 기반 rule로 라우팅
        logger.warning(f"Router JSON 파싱 실패: {e} — rule-based fallback 적용")
        rewritten_query = last_message
        route = "tools" if _needs_tools(last_message) else "learn"
        active_agents = [] if route == "tools" else ["retrieval", "socratic"]
        subtask = last_message

    # ── 5. 후처리 ────────────────────────────────────────────────

    # 첫 턴이거나 재작성 결과가 비었으면 원문 사용
    if is_first_turn or not rewritten_query:
        rewritten_query = last_message

    # route 값 정규화: 예상치 못한 값은 learn으로 폴백
    if route not in ("learn", "chat", "tools", "escape"):
        logger.warning(f"알 수 없는 route '{route}' — 'learn'으로 폴백")
        route = "learn"

    # tools route는 retrieval/socratic 없이 바로 도구 실행
    if route == "tools":
        active_agents = []

    # escape route: pending_quiz는 composer fast-path이므로 socratic만 실행
    # (pending_quiz 없을 때만 socratic이 <answer>를 생성해야 함)
    if route == "escape":
        active_agents = ["socratic"]

    # learn route이면 retrieval/socratic을 기본으로 포함
    if route == "learn" and not active_agents:
        active_agents = ["retrieval", "socratic"]

    # rule-based override: LLM이 learn으로 판단했어도 도구 키워드 있으면 tools로 전환
    if route == "learn" and _needs_tools(last_message):
        if "tools" not in active_agents:
            active_agents = active_agents + ["tools"]
            logger.info("도구 키워드 감지 — active_agents에 'tools' 추가.")

    # state 업데이트
    state["rewritten_query"] = rewritten_query
    state["route"] = route
    state["active_agents"] = active_agents
    state["subtask"] = subtask

    logger.info(f"route={route} | active_agents={active_agents} | query='{rewritten_query}'")
    _log_trace(
        step="Router",
        purpose="쿼리 재작성 + 라우팅 + 에이전트 선택 (단일 LLM 호출)",
        inputs={
            "Original": last_message,
            "History turns": len(messages) - 1,
            "Frustration": frustration_count,
        },
        prompt_details={"History": history_text, "Prompt": prompt},
        response=response_str,
        decision={
            "rewritten_query": rewritten_query,
            "route": route,
            "active_agents": active_agents,
            "subtask": subtask,
        },
    )
    return state
