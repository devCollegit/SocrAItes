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
from src.tools.learning_tools import search_web
from src.agent.helpers import _log_trace, _is_summary_request

logger = logging.getLogger("SocrAItes.Agent")


def _count_turns_since_topic_start(history: list, last_msg: str) -> int:
    """LLM으로 현재 메시지가 새 토픽인지 판단 후 해당 토픽의 Tutor 반문 횟수를 반환."""
    if len(history) <= 1:
        return 0

    # 최근 6개 메시지만 사용해 비용 최소화
    recent = history[-6:]
    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in recent[:-1]  # 마지막 유저 메시지 제외
    )

    prompt = (
        "아래는 최근 대화 히스토리와 학생의 새 메시지야.\n\n"
        f"대화 히스토리:\n{history_text}\n\n"
        f"학생의 새 메시지: \"{last_msg}\"\n\n"
        "학생의 새 메시지가 이전 대화와 다른 새로운 학습 토픽을 시작하는가?\n"
        "새 토픽이면 'NEW', 기존 대화의 연속이거나 잡담이면 'SAME'만 출력해."
    )
    try:
        result = _get_content(llm.invoke([HumanMessage(content=prompt)])).strip().upper()
        is_new_topic = "NEW" in result
    except Exception:
        is_new_topic = False

    if is_new_topic:
        return 0

    # 기존 토픽 연속 — 현재 히스토리에서 Tutor 반문 횟수 카운트
    return sum(1 for m in history if m["role"] == "assistant")

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

SOCRATIC_PROMPT = """You are SocrAItes, a world-class Socratic tutor.

Your response MUST always contain three clearly separated sections in this exact format:

<answer>
[강의 자료에 근거한 정확한 직접 정답. 학생이 이해하기 쉽게 한국어로 작성.]
</answer>
<feedback>
[학생의 직전 답변에 대한 피드백. 정답/부분정답/오답 여부를 한 문장으로 명확히 밝힌다.
학생의 직전 발화가 없거나 처음 질문인 경우 반드시 아무것도 쓰지 말고 태그만 남긴다: <feedback></feedback>]
</feedback>
<question>
[다음 소크라테스식 질문 한 개. feedback 없이 질문만 작성.]
</question>

<answer> 작성 규칙:
- 강의 자료에 근거한 정확한 정답을 먼저 완성한다 (내부 참고용이지만 정확해야 함).
- 요약 요청 시: 강의 자료를 구조적으로 정리한 상세 요약을 작성한다.

<question> 작성 규칙:
- 질문은 반드시 하나만. 여러 개 금지.
- 학생이 이미 알 법한 더 단순한 선행 개념에서 출발한다. 모르는 개념을 바로 묻지 말고 거슬러 올라가서 시작한다.
  예) LoRA를 모른다면 → "파라미터를 많이 학습하면 어떤 문제가 생길 것 같아요?"
- 예/아니오로 끝나는 질문 금지 — 학생이 자기 말로 설명하도록 유도하는 형태로 묻는다.
- 학생 답변이 있으면 반드시 첫 문장에서 맞고 틀림을 명확히 피드백한 뒤 질문으로 넘어간다.
  이전 질문을 무시하고 바로 새 질문을 던지는 것은 금지.
  ✓ 정답 → "정확해요! [핵심 개념명]을 잘 짚었어요. 그렇다면..." 으로 이어서 심화
  ✓ 부분 정답 → "좋은 접근이에요. [맞는 부분]은 맞는데, [빠진 부분]도 생각해보면..."
  ✓ 틀린 방향 → "흥미로운 생각이에요. 그런데 [부분]을 다시 생각해보면..." 으로 재고 유도
  ✓ "모르겠어", "모르겠다", "몰라" 등 모른다는 표현 → 질문을 절반 이하로 쪼개거나,
    정답의 첫 번째 단서(가장 쉬운 부분)만 힌트로 주고 그것에 대해서만 묻는다.
    같은 질문을 다른 말로 반복하는 것은 금지.
  ✓ 완전히 엉뚱한 답 → 질문을 더 작게 쪼개서 다시 물어본다
- 학생 답변의 어느 부분이 맞고 어느 부분이 부족한지 내부적으로 판단한 뒤 질문을 구성하라.
- 질문은 반드시 정답 방향으로 한 발짝 가까워지도록 설계하라. 학생이 그 질문에 답하면
  자연스럽게 정답에 가까워져야 한다. 막연히 "생각해보세요" 식의 질문은 금지.
- 절대 금지: 학생이 실제로 말하지 않은 내용을 "~라고 했는데", "~라고 하셨는데" 형태로 인용하지 말 것. 오직 학생이 직접 입력한 텍스트만 참조할 것.
- frustration_level >= 2이면: 질문 대신 구체적인 예시를 하나 들고 "이 예시에서 X는 어떤 역할을 하는 것 같나요?" 식으로 범위를 좁혀서 묻는다.
- 한 개념이 충분히 이해된 후에만 다음 개념으로 넘어간다. 넘어갈 때는 "방금 이해한 X와 연결해서 생각해보면..." 으로 브리지를 만든다.

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

Socratic depth: {depth} — 아래 모드에 따라 반문 횟수와 깊이를 엄격하게 지켜라.
{depth_guide}
Frustration level: {frustration_level}
Reviewer feedback (non-empty면 개선): "{eval_feedback}"

Lecture context:
---
{context}
---

Conversation history:
{history}"""

# ──────────────────────────────────────────────────────────────────────────────
# 깊이별 모드 지침
# ──────────────────────────────────────────────────────────────────────────────

_DEPTH_INSTRUCTIONS: dict = {
    0: (
        "Light 모드 — 힌트 중심, 빠른 개념 확인:\n"
        "- 가장 쉬운 선행 개념 한 가지만 묻는다.\n"
        "- 학생이 부분적으로라도 이해를 보이면 즉시 칭찬하고 정답을 알려준다.\n"
        "- 질문은 단답형 또는 짧은 서술형으로 구성한다.\n"
        "- 학생이 막히면 즉시 쉬운 비유·예시를 먼저 제공하고 확인 질문으로 마무리.\n"
        "- 빠른 피드백과 친절한 어조를 유지한다."
    ),
    1: (
        "Standard 모드 — 단계적 개념 탐구:\n"
        "- 개념 이해를 단계적으로 쌓아가며 최대 4회 반문한다.\n"
        "- 연관 개념을 연결하는 질문으로 이해 깊이를 확인한다.\n"
        "- 부분적 이해는 인정하고, 빠진 부분만 채우도록 유도한다.\n"
        "- 학생이 2회 연속 모른다고 하면 힌트와 함께 질문 범위를 좁힌다."
    ),
    2: (
        "Deep 모드 — 순수 소크라테스식 심층 탐구:\n"
        "- 전제 자체를 의심하는 질문, 가설 설정, 개념 간 연결을 유도한다.\n"
        "- '왜 그렇게 생각하나요?', '그 가정이 틀린다면 어떻게 되나요?' 형태의 메타 질문 권장.\n"
        "- 강의 자료를 넘어 실제 사례·최신 연구·실무 응용까지 확장한다.\n"
        "- 웹 검색 결과가 컨텍스트에 포함된 경우 적극 활용해 심층 질문을 구성한다.\n"
        "- 개념 간 상위 연결(메타 수준)을 묻는 질문을 적극 활용한다."
    ),
}

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

    # ── Deep 모드: 웹 검색으로 컨텍스트 보강 ─────────────────────
    if depth == 2 and last_msg:
        _search_decision_prompt = (
            f"강의 자료:\n{context[:1000]}\n\n"
            f"학생 질문: {last_msg}\n\n"
            "강의 자료만으로 학생 질문에 충분히 답할 수 있나?\n"
            "충분하면 'NO_SEARCH', 검색이 필요하면 'SEARCH: <검색어>'만 출력해. 다른 말은 하지 마."
        )
        try:
            _decision = _get_content(
                llm.invoke([HumanMessage(content=_search_decision_prompt)])
            ).strip()
            if _decision.upper().startswith("SEARCH:"):
                _query = _decision[7:].strip()
                if _query:
                    _result = search_web({"query": _query, "max_results": 3})
                    if _result.get("status") == "success" and _result.get("results"):
                        _snippets = "\n\n".join(
                            f"[{r['title']}]\n{r['content'][:500]}"
                            for r in _result["results"]
                        )
                        context += f"\n\n[웹 검색 결과: '{_query}']\n{_snippets}"
                        logger.info("Deep 모드 웹 검색 완료: query=%s", _query)
        except Exception as _e:
            logger.warning("Deep 모드 웹 검색 실패: %s", _e)

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

    # ── LLM 기반 토픽 변경 감지 후 반문 횟수 산출 ───────────────
    tutor_question_count = _count_turns_since_topic_start(history, last_msg)
    _MAX_TURNS = {0: 2, 1: 4, 2: 6}
    max_turns = _MAX_TURNS.get(depth, 4)
    turns_left = max_turns - tutor_question_count

    if turns_left <= 0:
        depth_guide = (
            f"[반문 한도 초과] 이미 {tutor_question_count}회 반문했다. "
            "지금은 반드시 직접 설명으로 전환해야 한다. "
            "<question>에 질문 대신 핵심 개념을 명확히 설명하고 학생이 이해했는지 확인하는 문장으로 마무리하라."
        )
    else:
        _MODE_NAMES = {0: "Light", 1: "Standard", 2: "Deep"}
        _mode_instruction = _DEPTH_INSTRUCTIONS.get(depth, _DEPTH_INSTRUCTIONS[1])
        depth_guide = (
            f"{_mode_instruction}\n\n"
            f"진행 상황: {_MODE_NAMES.get(depth, 'Standard')} 모드 — 최대 {max_turns}회 반문.\n"
            f"현재까지 반문 횟수: {tutor_question_count}회 / 남은 횟수: {turns_left}회.\n"
            f"남은 횟수가 0이 되면 다음 응답에서 반드시 직접 설명으로 전환한다."
        )

    # ── 대화 히스토리 텍스트 포맷 ─────────────────────────────────
    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Tutor'}: {m['content']}"
        for m in history
    )

    # ── 반문 한도 초과 여부를 state에 기록 (composer에서 처리) ───
    state["force_explain"] = turns_left <= 0

    # ── 시스템 프롬프트 조합 ─────────────────────────────────────
    system_content = summary_override + SOCRATIC_PROMPT.format(
        depth=depth,
        depth_guide=depth_guide,
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
