"""Sub Agent — Socratic Agent: 소크라테스식 질문과 힌트를 생성.

기존 dialogue.py 로직과 동일하되 필드명만 변경:
- sub_agents        →  active_agents   (스킵 조건 체크)
- socratic_response →  tutor_response  (출력 필드)
- contextualized_query → rewritten_query

사용자 프로필 기반 개인화:
- 학습 스타일 (practical/conceptual/concise)
- 선호 어조 (encouraging/strict/academic)
- 학업 배경 (대학원생 등)
- AI 노트, 강점/약점 요약
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import _log_trace, _is_summary_request

logger = logging.getLogger("SocrAItes.Agent")

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

SOCRATIC_PROMPT = """You are SocrAItes, a world-class Socratic tutor.

Your response MUST always contain two clearly separated sections in this exact format:

<answer>
[강의 자료에 근거한 정확한 직접 정답. 학생이 이해하기 쉽게 한국어로 작성.]
</answer>
<question>
[학생이 스스로 answer에 도달하도록 유도하는 소크라테스식 질문/힌트. 한국어로 작성.]
</question>

<answer> 작성 규칙:
- 강의 자료에 근거한 정확한 정답을 먼저 완성한다 (내부 참고용이지만 정확해야 함).
- 요약 요청 시: 강의 자료를 구조적으로 정리한 상세 요약을 작성한다.

<question> 작성 규칙:
- 직접 답을 주지 말고 힌트나 맥락(1-2문장)을 제공한 뒤 구체적인 질문을 한다.
- frustration_level >= 2이면 더 자세한 힌트를 제공한다.
- 모호한 의견 질문 금지 — 구체적인 메커니즘, 차이점, 하위 개념에 대해 질문한다.

Progressive Learning (필수):
- 이미 충분히 논의된 개념은 다시 묻지 말고 다음 논리적 개념으로 넘어간다.
- 학생이 잘 답하면 짧게 긍정 피드백 후 다음 토픽을 소개한다.

Personalization:
- Learning Style: '{learning_style}' (practical=실사례, conceptual=이론, concise=간결)
- Preferred Tone: '{preferred_tone}' (encouraging=격려, strict=엄격, academic=학술)
- Academic Background: '{academic_background}'
- AI Notes: {profile_notes}
- Strengths: "{strengths_summary}"
- Weaknesses: "{weaknesses_summary}"

Socratic depth: {depth} (0=Light, 1=Standard, 2=Deep)
Frustration level: {frustration_level}
Reviewer feedback (non-empty면 개선): "{eval_feedback}"

Lecture context:
---
{context}
---

Conversation history:
{history}"""

# ──────────────────────────────────────────────────────────────────────────────
# 메인 노드 함수
# ──────────────────────────────────────────────────────────────────────────────

def socratic_agent(state: AgentState) -> AgentState:
    """소크라테스식 질문/힌트를 생성한다. 'socratic'이 active_agents에 없으면 스킵.

    처리 순서:
    1. active_agents 체크 — "socratic" 없으면 tutor_response="" 로 스킵
    2. 사용자 프로필 로드 (state 우선, DB 폴백)
    3. 요약 요청 감지 → summary_override 주입
    4. LLM 호출로 소크라테스 응답 생성
    5. tutor_response 저장
    """
    logger.info("--- [Socratic Agent] Step ---")

    # active_agents에 "socratic"이 없으면 이 턴은 소크라테스 대화 불필요
    if "socratic" not in state.get("active_agents", ["socratic"]):
        state["tutor_response"] = ""
        _log_trace(
            step="SocraticAgent", purpose="스킵.", decision="active_agents에 'socratic' 없음."
        )
        return state

    # ── 필요 상태값 추출 ─────────────────────────────────────────
    docs = state.get("retrieved_docs", [])
    history = state.get("messages", [])
    depth = state.get("socratic_depth", 1)
    frustration_level = state.get("frustration_level", 0)
    eval_feedback = state.get("evaluation", {}).get("feedback", "")

    # ── 사용자 ID 조회 ────────────────────────────────────────────
    # session_id → DB에서 user_id 조회 (실패 시 "default" 사용)
    session_id = state.get("session_id")
    user_id = "default"
    if session_id:
        try:
            from src.db.database import get_session
            sess = get_session(session_id)
            if sess:
                user_id = sess.get("user_id", "default")
        except Exception as e:
            logger.warning(f"세션 조회 실패: {e}")

    # ── 사용자 프로필 로드 ────────────────────────────────────────
    # state["user_profile"]이 비어 있으면 DB에서 조회
    user_profile = state.get("user_profile", {})
    if not user_profile:
        try:
            from src.db.database import get_user_profile
            user_profile = get_user_profile(user_id)
        except Exception:
            user_profile = {}

    # 프로필 필드 추출 (기본값 포함)
    learning_style = user_profile.get("learning_style", "conceptual")
    preferred_tone = user_profile.get("preferred_tone", "encouraging")
    academic_background = user_profile.get("academic_background", "대학원생")
    profile_notes = user_profile.get("notes", "None")
    strengths_summary = user_profile.get("strengths_summary", "") or "No recorded strengths yet."
    weaknesses_summary = (
        user_profile.get("weaknesses_summary", "") or "No unresolved weaknesses recorded yet."
    )

    # ── 강의 자료 컨텍스트 조합 ───────────────────────────────────
    context = "\n".join(d["text"] for d in docs) if docs else "강의 자료 없음."
    last_msg = history[-1]["content"] if history else ""

    # ── 요약 요청 감지 → 프롬프트 앞단에 오버라이드 삽입 ─────────
    # "요약해줘", "정리해줘" 등의 요청 시 구조화된 요약 먼저 제공하도록 지시
    is_summary = _is_summary_request(last_msg) or _is_summary_request(
        state.get("rewritten_query", "")
    )
    summary_override = (
        "IMPORTANT: The student requested a summary. "
        "Provide a comprehensive, structured, detailed summary of the lecture context first, "
        "then close with one concrete Socratic question.\n"
        if is_summary
        else ""
    )

    # ── 대화 히스토리 텍스트 포맷 ─────────────────────────────────
    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in history
    )

    # ── 시스템 프롬프트 조합 ─────────────────────────────────────
    system_content = summary_override + SOCRATIC_PROMPT.format(
        depth=depth,
        frustration_level=frustration_level,
        eval_feedback=eval_feedback,
        learning_style=learning_style,
        preferred_tone=preferred_tone,
        academic_background=academic_background,
        profile_notes=profile_notes,
        strengths_summary=strengths_summary,
        weaknesses_summary=weaknesses_summary,
        context=context,
        history=history_text,
    )

    # ── LLM 호출 ─────────────────────────────────────────────────
    response = llm.invoke([SystemMessage(content=system_content), HumanMessage(content=last_msg)])
    tutor_response = _get_content(response)
    state["tutor_response"] = tutor_response

    logger.info(f"소크라테스 응답 생성 완료 (len={len(tutor_response)}).")
    _log_trace(
        step="SocraticAgent",
        purpose="강의 자료 기반 소크라테스식 질문/힌트 생성.",
        inputs={"Depth": depth, "Frustration": frustration_level, "Docs": len(docs)},
        prompt_details=system_content,
        response=tutor_response,
        decision="tutor_response 저장 완료.",
    )
    return state
