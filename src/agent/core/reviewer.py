"""Core Agent — Reviewer: 4축 품질 검사 + 조건부 실행 헬퍼.

기존 evaluator.py 로직을 그대로 유지하되 이름만 변경하고
`should_review(state)` 헬퍼를 추가해 조건부 실행을 명시적으로 처리한다.

4축 평가:
- socratic    : 소크라테스식 질문 (직접 답변 회피, 5=순수 소크라테스)
- grounding   : 강의 자료 기반 여부 (5=명확히 인용)
- encouragement: 따뜻하고 격려적인 어조 (5=매우 격려적)
- clarity     : 명확하고 체계적인 구성 (5=매우 명확)

통과 기준: 모든 점수 >= 3
실패 시: retry_count 증가 + 한국어 개선 피드백 생성
"""

import os
import re
import json
import logging

from langchain_core.messages import HumanMessage

from src.agent.state import AgentState
from src.agent.llm import llm, llm_strong, _get_content
from src.agent.helpers import _log_trace

logger = logging.getLogger("SocrAItes.Agent")

# ──────────────────────────────────────────────────────────────────────────────
# 프롬프트
# ──────────────────────────────────────────────────────────────────────────────

REVIEWER_PROMPT = """You are the Reviewer for SocrAItes. Evaluate the draft response on 4 axes.

Student's Latest Message:
"{last_msg}"

Draft Response:
"{draft}"

Internal Tutor Answer (Hidden from student, for your reference only):
"{tutor_answer}"
* Note: This answer is purely for internal reference to understand the tutor's goal. Do NOT penalize the tutor for providing direct explanations here. 
Furthermore, you should explicitly check if the "Draft Response" utilizes the prior knowledge and concepts from this internal answer to provide helpful hints and scaffolding for the student's latest message. Providing such hints is highly encouraged and should positively impact the socratic score.

Lecture context (excerpt):
"{context_excerpt}"

Axes (score 1-5 each):
{socratic_criteria}
2. grounding: 제공된 강의 자료에 기반하는가 (5=명확히 강의 자료 인용, 1=전혀 관련 없음)
3. encouragement: 따뜻하고 지지적인 어조인가 (5=매우 격려적, 1=냉담함)
4. clarity: 명확하고 잘 구조화되어 있는가 (5=매우 명확, 1=혼란스러움)

Pass criteria: 모든 점수 >= 3.
점수 < 3인 항목이 있으면 pass=false로 설정하고 한국어로 구체적이고 실행 가능한 개선 제안을 작성하세요.

Respond ONLY in JSON:
{{
  "scores": {{"socratic": <int>, "grounding": <int>, "encouragement": <int>, "clarity": <int>}},
  "pass": <bool>,
  "feedback": "<한국어 개선 제안, 통과 시 빈 문자열>"
}}"""

# ──────────────────────────────────────────────────────────────────────────────
# 조건부 실행 헬퍼
# ──────────────────────────────────────────────────────────────────────────────

def should_review(state: AgentState) -> bool:
    """reviewer 노드를 실행해야 하는지 판단한다.

    reviewer는 다음 조건을 모두 충족할 때만 실행한다:
    1. route == "learn" — tools/chat route는 항상 스킵
    2. 아래 중 하나라도 해당:
       a. retry_count > 0        : 이미 한 번 이상 재시도 중
       b. frustration_level >= 2 : 학생이 많이 좌절하고 있음
       c. "tools" not in active_agents : 도구 없이 순수 소크라테스 응답만 생성됨
    """
    # tools/chat route는 reviewer 불필요
    if state.get("route") != "learn":
        return False

    retry_count = state.get("retry_count", 0)
    frustration_level = state.get("frustration_level", 0)
    active_agents = state.get("active_agents", [])

    # 재시도 중이거나, 좌절 수준이 높거나, 도구 없이 소크라테스 응답만 생성된 경우 검토
    return (
        retry_count > 0
        or frustration_level >= 2
        or "tools" not in active_agents
    )


# ──────────────────────────────────────────────────────────────────────────────
# 메인 노드 함수
# ──────────────────────────────────────────────────────────────────────────────

def reviewer(state: AgentState) -> AgentState:
    """4축 품질 검사를 수행한다. 실패 시 retry_count를 증가시켜 재시도를 유발한다."""
    logger.info("--- [Reviewer] Quality Check Step ---")
    # 평가 대상: composer가 생성한 최종 응답
    draft = state.get("response", "")
    docs = state.get("retrieved_docs", [])
    # 강의 자료 중 앞 2개 청크의 텍스트를 컨텍스트로 사용 (최대 400자)
    context_excerpt = " ".join(d["text"] for d in docs[:2])[:400] if docs else ""

    # 강제 설명 모드 또는 요약 요청인지 확인하여 소크라테스 평가 기준 완화
    force_explain = state.get("force_explain", False)
    from src.agent.helpers import _is_summary_request
    messages = state.get("messages", [])
    last_msg = messages[-1]["content"] if messages else ""
    is_summary = _is_summary_request(last_msg) or _is_summary_request(state.get("rewritten_query", ""))

    if force_explain or is_summary:
        socratic_criteria = "1. socratic: [예외 상황] 강제 설명 모드 또는 요약 요청이므로 직접적인 설명이나 긴 요약이 적극 허용됩니다. (무조건 5점 부여)"
    else:
        socratic_criteria = "1. socratic: 직접 답변을 피하고 소크라테스식 질문을 사용하는가? (단, 학생의 직전 답변에 대한 구체적이고 친절한 피드백은 적극 권장됨) (5=적절한 피드백과 함께 순수 소크라테스 질문, 1=완전 직접 답변)"

    tutor_response = state.get("tutor_response", "")
    match = re.search(r"<answer>(.*?)</answer>", tutor_response, re.DOTALL)
    tutor_answer = match.group(1).strip() if match else "없음"

    prompt = REVIEWER_PROMPT.format(
        last_msg=last_msg,
        draft=draft, 
        tutor_answer=tutor_answer,
        context_excerpt=context_excerpt, 
        socratic_criteria=socratic_criteria
    )

    try:
        if os.getenv("OPENAI_API_KEY"):
            response = llm_strong.bind(response_format={"type": "json_object"}).invoke(
                [HumanMessage(content=prompt)]
            )
        else:
            response = llm_strong.invoke([HumanMessage(content=prompt)])
        parsed = json.loads(_get_content(response).strip())
        scores = parsed.get("scores", {})
        passed = parsed.get("pass", True)
        feedback = parsed.get("feedback", "")
    except Exception as e:
        # 파싱 실패 시 통과 처리 (무한 루프 방지)
        logger.warning(f"Reviewer 파싱 실패: {e}. 기본값(통과)으로 처리.")
        scores, passed, feedback = {}, True, ""

    state["evaluation"] = {"pass": passed, "feedback": feedback, "scores": scores}

    # 실패 시 retry_count를 증가시켜 graph.py의 route_after_reviewer가 재시도를 결정
    if not passed:
        state["retry_count"] = state.get("retry_count", 0) + 1

    logger.info(
        f"Review: pass={passed} scores={scores} retry={state.get('retry_count', 0)}"
    )
    _log_trace(
        step="Reviewer",
        purpose="4축 품질 검사: Socratic, grounding, encouragement, clarity.",
        inputs={"Draft length": len(draft), "Docs": len(docs)},
        prompt_details=prompt,
        response=state["evaluation"],
        decision=f"Pass={passed} | Retries={state.get('retry_count', 0)}",
    )
    return state
