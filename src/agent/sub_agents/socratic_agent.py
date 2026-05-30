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
from src.agent.llm import llm, llm_strong, _get_content
from src.tools.learning_tools import search_web
from src.agent.helpers import _log_trace, _is_summary_request

logger = logging.getLogger("SocrAItes.Agent")


def _is_concept_question(text: str) -> bool:
    """'X가 뭐야?', 'X를 설명해줘' 등 개념 자체를 묻는 질문인지 판단한다."""
    keywords = ["가 뭐야", "이 뭐야", "은 뭐야", "는 뭐야",
                "가 뭔가요", "이 뭔가요", "란 무엇", "이란 무엇",
                "를 설명해", "을 설명해", "에 대해 설명", "이 뭐지", "가 뭐지"]
    clean = text.replace(" ", "")
    return any(k.replace(" ", "") in clean for k in keywords)


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
[학생의 직전 답변에 대한 평가를 한 문장으로만 작성한다.
⚠️ 절대 금지: "그렇다면", "그럼", 질문 문장. 오직 정답/부분정답/오답 평가만.
주제 전환이나 다음 개념 브리지는 반드시 <question>에 작성한다.
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
- 【핵심 규칙】 학생이 "X가 뭐야?", "X를 설명해줘" 처럼 개념 자체를 모르는 질문을 한 경우:
  ① 먼저 내부적으로 "X는 어떤 문제를 해결하기 위해 등장했는가?"를 Lecture context에서 파악한다.
     그 배경 문제로 학생을 이끌기 위해, 학생이 이미 알고 있을 선행 개념을 질문으로 구성한다.
     ⚠️ 금지: "X가 없었다면 어떤 문제가 있었을까요?" — X를 모르는 학생은 이 질문에 답할 수 없다.
     ⚠️ 질문은 반드시 학생이 이미 알고 있는 개념 수준으로 구성해야 한다.
     예) "LoRA가 뭐야?" → 내부 파악: "파라미터 전체 업데이트 시 학습 비용이 크다"
                        → 질문: "대형 모델을 fine-tuning할 때 모든 파라미터를 업데이트하면 어떤 문제가 생길 것 같아요?"
  ② 질문에 X의 이름, 구성요소, 세부 기법을 포함하는 것은 모두 금지.
     X의 내용을 담은 단어가 질문에 등장하는 것 자체가 정답 노출이다.
  ③ 학생이 배경 문제를 이해하면 그 다음 질문에서 반드시 해결 아이디어 방향으로 이동한다.
     ⚠️ 금지: 이미 확인된 문제에서 "또 다른 문제"를 추가로 묻는 횡이동.
     예) 학생이 "시간이 오래 걸린다"고 맞혔을 때:
       ✗ "파라미터를 많이 학습하면 또 어떤 문제가 있을까요?" (횡이동)
       ✓ "그 문제를 줄이려면 어떤 아이디어가 있을까요?" (해결 방향 전진)
- 예/아니오로 끝나는 질문 금지 — 학생이 자기 말로 설명하도록 유도하는 형태로 묻는다.
- 학생의 답이 질문의 초점과 다르더라도 사실 자체가 맞으면 반드시 정답/부분정답으로 인정한다.
  예) "초기화 방식은?" 질문에 "선형 변환이요" →
    ✗ "선형 변환보다는 초기화에 집중해야..." (사실을 틀리게 취급)
    ✓ "맞아요, 선형 변환이 맞아요. 그렇다면 A와 B가 어떤 값으로 시작하는지 알고 있나요?" (인정 후 유도)
- 질문에 등장하는 모든 개념은 학생이 이미 이해한 것이거나 현재 대화에서 다루고 있는 개념이어야 한다.
  학생이 아직 배우지 않은 새 개념을 질문에 먼저 도입하는 것은 금지.
- 학생 답변이 있으면 반드시 첫 문장에서 맞고 틀림을 명확히 피드백한 뒤 질문으로 넘어간다.
  이전 질문을 무시하고 바로 새 질문을 던지는 것은 금지.
  ✓ 정답 → "정확해요! [핵심 개념명]을 잘 짚었어요. 그렇다면..." 으로 이어서 심화
  ✓ 부분 정답 → "좋은 접근이에요. [맞는 부분]은 맞는데, [빠진 부분]도 생각해보면..."
  ✓ 틀린 방향 → "흥미로운 생각이에요. 그런데 [부분]을 다시 생각해보면..." 으로 재고 유도
  ✓ "모르겠어", "모르겠다", "몰라" 등 모른다는 표현 →
    ⚠️ 절대 금지: 다른 주제·새로운 질문으로 넘어가는 것.
    ⚠️ 절대 금지: "이 방법의 이름은?", "어떤 것이 있을까요?" 처럼 학생이 모른다고 밝힌 내용의 답을 그대로 다시 요구하는 질문.
    ⚠️ 절대 금지: "힌트를 드릴게요" 같은 예고만 쓰고 실제 힌트 내용 없이 질문만 하는 것.
    반드시 다음 형식을 지킨다:
      "힌트: [강의 자료에서 가져온 1~2문장의 구체적 사실]. [그 힌트에 대한 쉬운 질문]?"
    예) "InstructGPT 해결 방법을 모르겠어" →
      ✗ "이 방법의 이름은 무엇일까요?" (답을 모르는 상태에서 이름 재요구)
      ✗ "어떤 접근 방법이 있을까요?" (주제 이탈)
      ✓ "힌트: InstructGPT는 사람의 피드백을 강화학습에 활용하는 RLHF 방법을 써요. 강화학습에서 사람의 피드백이 어떤 역할을 한다고 생각하나요?"
  ✓ 완전히 엉뚱한 답 → 질문을 더 작게 쪼개서 다시 물어본다
- 학생 답변의 어느 부분이 맞고 어느 부분이 부족한지 내부적으로 판단한 뒤 질문을 구성하라.
- 질문은 반드시 정답 방향으로 한 발짝 가까워지도록 설계하라. 학생이 그 질문에 답하면
  자연스럽게 정답에 가까워져야 한다. 막연히 "생각해보세요" 식의 질문은 금지.
- 학생이 문답 도중 새 개념을 질문한 경우("X랑 Y가 달라?", "X가 뭐야?" 등):
  <question>에 그 서브 질문에 대한 답을 한 문장으로 먼저 쓴 뒤, Socratic 질문으로 이어간다.
- 절대 금지: 학생이 실제로 말하지 않은 내용을 "~라고 했는데", "~라고 하셨는데" 형태로 인용하지 말 것. 오직 학생이 직접 입력한 텍스트만 참조할 것. 특히 학생이 "모르겠어"라고 한 직후에는 그 개념의 내용을 학생이 말한 것처럼 인용하는 것을 금지한다.
- frustration_level >= 2이면: 반드시 다음 형식을 지킨다:
    "예를 들어, [실제 상황·비유 1~2문장]. 이 상황에서 X는 어떤 역할을 할 것 같나요?"
  ⚠️ 절대 금지: "이 예시에서..."라는 표현만 쓰고 실제 예시 내용을 생략하는 것.
- 학생이 1회 정답/부분정답을 내면 즉시 다음 단계로 넘어간다. 같은 개념을 반복 확인하지 않는다.
  넘어갈 때는 "방금 이해한 X와 연결해서 생각해보면..." 으로 브리지를 만든다.

Progressive Learning (필수):
- 학생이 정답/부분정답을 낸 개념은 다시 묻지 않는다. 즉시 다음 논리적 단계로 이동한다.
- 특히 "X가 뭐야?" 질문에서: 배경 이해 후 반드시 X의 작동 방식·메커니즘으로 진행한다.
  배경만 반복하다 끝나는 것은 학생의 원래 질문에 답하지 않은 것이다.

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

    # ── 웹 검색으로 컨텍스트 보강 ────────────────────────────────
    # Deep 모드는 항상, 그 외 모드는 개념 정의 질문("X가 뭐야?")일 때만 실행
    _needs_search_check = depth == 2 or _is_concept_question(last_msg)
    if _needs_search_check and last_msg:
        _search_decision_prompt = (
            f"강의 자료:\n{context[:1000]}\n\n"
            f"학생 질문: {last_msg}\n\n"
            "소크라테스식 유도를 위해 '이 개념이 등장하기 전에 어떤 문제가 있었는가?"
            "(기존 방법의 한계, 불편함)'을 학생에게 먼저 물어야 한다.\n"
            "강의 자료에서 그 배경 문제(기존 한계)를 찾을 수 있나?\n"
            "찾을 수 있으면 'NO_SEARCH', 찾을 수 없으면 'SEARCH: <검색어>'만 출력해. 다른 말은 하지 마."
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
    response = llm_strong.invoke([SystemMessage(content=system_content), HumanMessage(content=last_msg)])
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
