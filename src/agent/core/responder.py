"""Core Agent — Responder: 캐주얼(비학습) 메시지에 대한 간단한 응답 생성.

기존 evaluator.py 내 direct_response 함수를 독립 모듈로 분리.
필드명: draft_answer → response

개인화: state["user_profile"]의 학습 스타일·어조·약점/강점 요약 등을
시스템 프롬프트에 반영하여 사용자 맞춤형 캐주얼 응답을 생성한다.
"""

import logging
from datetime import datetime, timezone, timedelta

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

# ── 어조 매핑 ────────────────────────────────────────────────────
_TONE_DESCRIPTION = {
    "encouraging": "따뜻하고 격려하는 어조",
    "strict": "간결하고 엄격한 어조",
    "academic": "학문적이고 정중한 어조",
}


def _build_responder_system(profile: dict) -> str:
    """user_profile을 반영한 개인화 시스템 프롬프트를 동적으로 생성한다.

    profile이 비어 있으면 기본 캐주얼 응답 프롬프트를 반환한다.
    """
    # 현재 시각 (KST) 을 프롬프트에 포함
    KST = timezone(timedelta(hours=9))
    now_kst = datetime.now(KST)
    time_str = now_kst.strftime("%Y-%m-%d %H:%M (%A)")  # e.g. "2026-05-31 23:46 (Sunday)"

    base = (
        "You are SocrAItes, a friendly academic assistant. "
        "Respond to the user's greeting or casual talk briefly in Korean.\n"
        f"현재 시각(한국 시간): {time_str}. "
        "시간대에 맞는 인사(좋은 아침/점심/저녁 등)를 사용하고, "
        "늦은 밤이면 건강을 챙기는 멘트를 자연스럽게 넣어주세요."
    )

    if not profile:
        return base

    # 개인화 조각들을 누적
    parts: list[str] = [base]

    # 1) 선호 어조 반영
    tone = profile.get("preferred_tone", "")
    if tone and tone in _TONE_DESCRIPTION:
        parts.append(f"사용자가 선호하는 어조: {_TONE_DESCRIPTION[tone]}로 답변하세요.")

    # 2) 학업 배경
    bg = profile.get("academic_background", "")
    if bg:
        parts.append(f"사용자의 학업 배경: {bg}. 이를 고려해 눈높이를 맞추세요.")

    # 3) 약점 요약 — 자연스럽게 격려 or 학습 유도 멘트에 활용
    weak_summary = profile.get("weaknesses_summary", "")
    if weak_summary:
        parts.append(
            f"사용자의 현재 약점 요약: {weak_summary}. "
            "적절한 경우 가볍게 격려하거나 관련 학습 동기를 부여하세요. "
            "단, 직접적으로 약점 목록을 나열하지는 마세요."
        )

    # 4) 강점 요약 — 칭찬·자신감 강화에 활용
    str_summary = profile.get("strengths_summary", "")
    if str_summary:
        parts.append(
            f"사용자의 강점 요약: {str_summary}. "
            "적절한 경우 자연스럽게 칭찬해 자신감을 높여주세요."
        )

    # 5) AI가 기록한 사용자 관련 메모
    notes = profile.get("notes", "")
    if notes:
        parts.append(f"참고 메모: {notes}")

    return "\n".join(parts)


def responder(state: AgentState) -> AgentState:
    """chat route에서 호출. 인사/잡담에 짧고 친근한 한국어 응답을 생성한다.

    state["user_profile"]에 담긴 학습 스타일, 어조, 약점/강점 요약을
    시스템 프롬프트에 반영해 개인화된 응답을 만든다.

    학습 내용을 다루지 않으므로 retrieval, socratic, reviewer 노드를 거치지 않는다.
    응답은 state["response"]에 저장되고 그래프는 END로 종료된다.
    """
    logger.info("--- [Responder] Step ---")
    last_msg = state["messages"][-1]["content"]
    profile = state.get("user_profile", {})

    # 프로필 기반 개인화 시스템 프롬프트 생성
    system_prompt = _build_responder_system(profile)
    logger.debug("[Responder] profile keys=%s", list(profile.keys()) if profile else "empty")

    # LLM에 시스템 + 사용자 메시지를 직접 전달해 간결한 응답을 얻는다
    state["response"] = _get_content(
        llm.invoke([SystemMessage(content=system_prompt), HumanMessage(content=last_msg)])
    )

    _log_trace(
        step="Responder",
        purpose="인사/잡담에 대한 개인화 한국어 응답 생성.",
        inputs={"Input": last_msg, "profile_tone": profile.get("preferred_tone", "N/A")},
        response=state["response"],
    )
    return state
