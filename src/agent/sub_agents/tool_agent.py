"""Sub Agent — Tool Agent: 학습 도구 호출 + 백그라운드 프로필 업데이트.

기존 diagnosis.py를 개선한 버전:
1. route == "tools"  : 명시적 도구 요청 처리 (retrieval/socratic 없이 바로 실행)
2. route == "learn"  : "tools" in active_agents일 때만 실행, 아니면 스킵
3. 도구 호출 후 threading.Thread(daemon=True)로 사용자 프로필 백그라운드 업데이트

필드명 변경:
- plan            →  subtask
- sub_agents      →  active_agents
- diagnosis_result →  tool_result
"""

import logging
import threading
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content, _extract_tool_calls, _run_tool_calls, LANGCHAIN_TOOLS
from src.tools.learning_tools import LANGCHAIN_TOOLS_DEEP
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

# "learn" route 내에서 진단/저장/퀴즈 등 도구를 호출할 때 사용하는 상세 프롬프트
LEARN_TOOL_PROMPT = """You are the Tool Agent for SocrAItes.

대화를 분석해 적절한 학습 도구를 호출하세요.

📙 도구 호출 규칙:

1. generate_quiz: 학생이 명시적으로 퀴즈/연습문제를 요청할 때.
   예: "문제 내달라", "연습해보고 싶어", "퀴즈 풀어볼까"

2. save_weakness: 아래 패턴이 감지되면 반드시 호출:
   ✓ 개념에 대해 질문함 ("X가 뭐예요?", "X 왜 필요해?")
   ✓ 이해 못 함을 표현 ("모르겠어요", "이해가 안 돼")
   ✓ 잘못된 설명이나 오개념 제시
   ✓ 두 개념을 혼동 ("A와 B 차이가 뭐예요?")
   ✓ 불완전한 이해 또는 틀린 멘탈 모델
   → 개념과 세부 내용을 대화에서 추론. 심각도 1-5 매핑.
   → 누락하지 말 것 — 학습 간극이 조금이라도 보이면 바로 호출.

3. save_strength: 학생이 개념 숙달을 명확히 보여줄 때:
   ✓ 개념을 올바르고 완전하게 설명
   ✓ 힌트 없이 소크라테스 질문에 정확히 답변
   ✓ 이전 오개념을 스스로 수정

4. schedule_review: 복습 일정을 원할 때 ("복습 일정 잡고 싶어").
   datetime은 선택 (기본: 지금으로부터 7일 후). 최근 주제에서 설명을 추론.

IMPORTANT:
- 도구를 적극적으로 호출하세요. 추가 질문 없이 아는 정보로 바로 호출.
- save_weakness: 학습 간극이 조금이라도 보이면 즉시 호출.
- 선택적 파라미터 누락은 괜찮음 — 기본값 사용.
- 도구가 필요 없으면 "No tools needed."로 응답.

Current subtask: {subtask}
Frustration level: {frustration_level}
Current time (KST): {current_time}"""

# "tools" route에서 명시적 도구 요청을 처리하는 간결한 프롬프트
DIRECT_TOOL_PROMPT = """You are the Tool Agent for SocrAItes.

학생의 명시적 도구 요청을 즉시 처리하세요. 질문 없이 바로 도구를 호출하세요.

요청: {last_message}
Subtask: {subtask}
Current time (KST): {current_time}"""


# ──────────────────────────────────────────────────────────────────────────────
# 메인 노드 함수
# ──────────────────────────────────────────────────────────────────────────────

def tool_agent(state: AgentState) -> AgentState:
    """도구를 호출하고 결과를 tool_result에 저장한다.

    route별 동작:
    - "tools" route : retrieval/socratic 없이 바로 명시적 도구 요청 처리
    - "learn" route : active_agents에 "tools"가 있을 때만 실행, 없으면 스킵

    도구 호출 후 백그라운드에서 프로필 업데이트 thread를 실행한다.
    """
    logger.info("--- [Tool Agent] Step ---")
    route = state.get("route", "learn")
    active_agents = state.get("active_agents", [])
    depth = state.get("socratic_depth", 1)

    # ── 실행 여부 판단 ────────────────────────────────────────────
    # escape: socratic_agent가 <answer>를 생성하므로 도구 호출 불필요
    # learn + tools 미포함: 이번 턴에 도구 요청 없음
    if route == "escape" or (route == "learn" and "tools" not in active_agents):
        state["tool_result"] = {}
        _log_trace(
            step="ToolAgent",
            purpose="스킵.",
            decision=f"route={route}, active_agents={active_agents}.",
        )
        return state

    # ── 필요 상태값 추출 ─────────────────────────────────────────
    subtask = state.get("subtask", "")
    frustration_level = state.get("frustration_level", 0)
    history = state.get("messages", [])
    last_msg = history[-1]["content"] if history else ""

    from datetime import datetime, timezone, timedelta
    KST = timezone(timedelta(hours=9))
    current_time_str = datetime.now(KST).strftime("%Y-%m-%d %H:%M (%A)")

    # ── route별 프롬프트 선택 ─────────────────────────────────────
    if route == "tools":
        # "tools" route: 명시적 도구 요청 — 간결한 프롬프트로 바로 실행
        system_content = DIRECT_TOOL_PROMPT.format(
            last_message=last_msg,
            subtask=subtask,
            current_time=current_time_str,
        )
    else:
        # "learn" route: 대화 분석 후 적절한 도구 선택 — 상세 프롬프트 사용
        system_content = LEARN_TOOL_PROMPT.format(
            subtask=subtask,
            frustration_level=frustration_level,
            current_time=current_time_str,
        )

    # ── LLM에 전달할 메시지 조합 ──────────────────────────────────
    messages_for_llm = [SystemMessage(content=system_content)]
    for m in history:
        if m["role"] == "user":
            messages_for_llm.append(HumanMessage(content=m["content"]))
        elif m["role"] == "assistant":
            messages_for_llm.append(AIMessage(content=m["content"]))

    # ── 도구 바인딩 LLM 호출 ─────────────────────────────────────
    _active_tools = LANGCHAIN_TOOLS_DEEP if depth == 2 else LANGCHAIN_TOOLS
    response = llm.bind_tools(_active_tools).invoke(messages_for_llm)
    analysis_text = _get_content(response)
    tool_calls = _extract_tool_calls(response)

    # ── session_id / user_id 주입 ─────────────────────────────────
    # save_weakness, save_strength 등 user 식별이 필요한 도구에 ID 주입
    session_id = state.get("session_id")
    user_profile = state.get("user_profile", {})
    user_id = user_profile.get("user_id", "default")

    if session_id and tool_calls:
        for tc in tool_calls:
            args = tc.get("args", {})
            if isinstance(args, dict):
                args["session_id"] = session_id
                if tc.get("name") in ["save_weakness", "save_strength"]:
                    args["user_id"] = user_id
                tc["args"] = args

    # ── 도구 실행 ─────────────────────────────────────────────────
    tool_results = _run_tool_calls(tool_calls) if tool_calls else []

    # ── tool_result 저장 ─────────────────────────────────────────
    state["tool_result"] = {
        "analysis": analysis_text,
        "tool_results": tool_results,
        "tool_calls_made": len(tool_calls),
    }
    state["tool_results"] = tool_results  # SSE 응답용 평탄화 목록

    logger.info(f"도구 에이전트 완료. 호출된 도구 수: {len(tool_calls)}.")
    _log_trace(
        step="ToolAgent",
        purpose="대화 분석 후 적절한 학습 도구 호출.",
        inputs={"Subtask": subtask, "Frustration": frustration_level, "Route": route},
        prompt_details=system_content,
        response=analysis_text,
        decision={"tool_calls": len(tool_calls), "results": tool_results},
    )

    return state
