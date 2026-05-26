"""Core Agent — Composer: 서브에이전트 출력을 조합해 최종 응답을 생성.

route별 처리:
- "learn"  : socratic_agent의 <question> 부분만 노출
- "escape" + pending_quiz : pending_quiz에서 정답 직접 추출 (LLM 0회)
- "escape" + no quiz : socratic_agent의 <answer> 부분만 노출
- "tools"  : 도구 실행 결과를 LLM으로 합성
- 퀴즈 채점: 사용자가 답안 제출 시 채점 결과 반환

우선순위:
1. 퀴즈 채점 (pending_quiz + 답안 제출)
2. escape + pending_quiz → 정답 직접 포맷 (LLM 없음)
3. escape + no quiz → tutor_response의 <answer> 추출
4. tool_result 내 도구 실행 결과 (퀴즈 생성, 약점/일정 저장)
5. tutor_response의 <question> 추출 (learn route 기본)
6. fallback LLM 생성
"""

import re
import json
import logging

from langchain_core.messages import HumanMessage, SystemMessage

from src.agent.state import AgentState
from src.agent.llm import llm, _get_content
from src.agent.helpers import (
    _log_trace,
    _is_quiz_answer,
    _is_last_message_quiz_prompt,
    _detect_frustration,
    _grade_quiz,
    _format_quiz_response,
)

logger = logging.getLogger("SocrAItes.Agent")


def _extract_answer(text: str) -> str:
    """socratic_agent 출력에서 <answer> 태그 내용을 추출한다."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_question(text: str) -> str:
    """socratic_agent 출력에서 <question> 태그 내용을 추출한다.
    태그가 없으면 전체 텍스트를 fallback으로 반환한다."""
    match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    return match.group(1).strip() if match else text


def _format_quiz_escape(pending_quiz: list) -> str:
    """pending_quiz의 정답을 규칙 기반으로 포맷한다. LLM 호출 없음."""
    lines = ["## 퀴즈 정답\n"]
    for i, item in enumerate(pending_quiz, 1):
        answer = item.get("answer", "").strip()
        options = item.get("options", [])
        question = item.get("question", "")

        # answer가 단일 알파벳(A~D)이면 인덱스로 변환해 선택지 텍스트 표시
        if len(answer) == 1 and answer.upper() in "ABCD":
            letter = answer.upper()
            idx = ord(letter) - ord("A")
            answer_text = options[idx] if options and 0 <= idx < len(options) else ""
            lines.append(f"{i}. {question}")
            lines.append(f"   정답: {letter}. {answer_text}\n")
        else:
            # LLM이 전체 텍스트로 반환한 경우 그대로 표시
            lines.append(f"{i}. {question}")
            lines.append(f"   정답: {answer}\n")
    return "\n".join(lines)


def composer(state: AgentState) -> AgentState:
    """서브에이전트 결과를 route에 따라 조합해 최종 응답(response)을 생성한다.

    처리 순서:
    1. [퀴즈 채점]  pending_quiz + 답안 형식 → 채점 후 반환
    2. [escape+퀴즈] escape route + pending_quiz → 정답 직접 추출 (LLM 없음)
    3. [escape]     escape route → tutor_response의 <answer> 추출
    4. [도구 결과]  tool_results → 퀴즈 포맷 또는 LLM 합성
    5. [learn]      tutor_response의 <question> 추출
    6. [폴백]       retrieved_docs 기반 LLM 생성
    """
    logger.info("--- [Composer] Synthesis Step ---")
    history = state.get("messages", [])
    last_msg = history[-1]["content"] if history else ""
    pending_quiz = state.get("pending_quiz", [])
    route = state.get("route", "learn")

    # ── 1. 퀴즈 채점 ─────────────────────────────────────────────
    # "1:A, 2:B" 형태의 답안 제출 감지 → 채점 결과 반환
    if pending_quiz and _is_quiz_answer(last_msg):
        state["response"] = _grade_quiz(last_msg, pending_quiz)
        state["pending_quiz"] = []
        state["tool_results"] = []
        logger.info("퀴즈 채점 완료.")
        _log_trace(step="Composer", purpose="퀴즈 답안 채점.", decision="채점 결과 → response.")
        return state

    # ── 2. escape + 퀴즈 중 → 정답 직접 추출 (LLM 0회) ──────────
    # pending_quiz에 이미 answer 필드가 있으므로 규칙 기반으로 포맷만 한다.
    if route == "escape" and pending_quiz:
        state["response"] = _format_quiz_escape(pending_quiz)
        state["pending_quiz"] = []   # 정답 공개 후 퀴즈 종료
        state["tool_results"] = []
        logger.info("escape route + pending_quiz → 퀴즈 정답 직접 포맷.")
        _log_trace(step="Composer", purpose="퀴즈 escape 정답 포맷 (LLM 없음).", decision="pending_quiz → response.")
        return state

    # ── 3. escape + 일반 질문 → <answer> 추출 ────────────────────
    # socratic_agent가 생성한 <answer> 태그 내용만 노출한다.
    tutor_response = state.get("tutor_response", "")
    if route == "escape" and tutor_response:
        answer = _extract_answer(tutor_response)
        # <answer> 태그가 없으면 전체 응답을 fallback으로 사용
        state["response"] = answer or tutor_response
        state["tool_results"] = []
        logger.info("escape route → <answer> 추출 완료.")
        _log_trace(step="Composer", purpose="escape: <answer> 추출.", decision="<answer> → response.")
        return state

    # ── 4. 도구 실행 결과 합성 ────────────────────────────────────
    tool_results = state.get("tool_result", {}).get("tool_results", [])
    if tool_results:
        # generate_quiz 결과 → 포맷팅된 퀴즈 메시지 + pending_quiz 설정
        quiz_message, quiz_items = _format_quiz_response(tool_results)
        if quiz_message:
            state["response"] = quiz_message
            state["tool_results"] = tool_results
            state["pending_quiz"] = quiz_items
            logger.info("generate_quiz 결과 포맷팅 완료.")
            _log_trace(step="Composer", purpose="퀴즈 포맷팅.", decision="pending_quiz 설정.")
            return state

        # 그 외 도구 결과 (save_weakness, save_strength, schedule_review) → LLM 합성
        synthesis_prompt = (
            "You are SocrAItes. 도구 실행 결과를 바탕으로 간결한 한국어 응답을 작성하세요.\n\n"
            f"Tool Results:\n{json.dumps(tool_results, ensure_ascii=False)}\n\n"
            "Rules:\n"
            "1. 약점/강점/일정이 저장되었으면 간단히 확인하고 소크라테스식 후속 질문 1개를 달아주세요.\n"
            "2. 간결하고 따뜻한 어조로 작성하세요.\n"
        )
        state["response"] = _get_content(
            llm.invoke([SystemMessage(content=synthesis_prompt), HumanMessage(content="최종 응답을 작성해줘.")])
        )
        state["tool_results"] = tool_results
        _log_trace(step="Composer", purpose="도구 결과 합성.", response=state["response"])
        return state

    # ── 5. learn route → <question> 추출 ─────────────────────────
    # socratic_agent 출력에서 <question> 태그 내용만 학생에게 노출한다.
    if tutor_response:
        state["response"] = _extract_question(tutor_response)
        state["tool_results"] = []
        logger.info("learn route → <question> 추출 완료.")
        _log_trace(step="Composer", purpose="learn: <question> 추출.", decision="<question> → response.")
        return state

    # ── 6. 폴백 LLM 생성 ─────────────────────────────────────────
    docs = state.get("retrieved_docs", [])
    context = "\n".join(d["text"] for d in docs) if docs else "강의 자료 없음."
    fallback_prompt = (
        f"You are SocrAItes. 아래 강의 자료를 바탕으로 간결한 소크라테스식 힌트와 질문을 "
        f"한국어로 생성하세요.\nContext: {context[:500]}\nStudent: {last_msg}"
    )
    state["response"] = _get_content(llm.invoke([HumanMessage(content=fallback_prompt)]))
    state["tool_results"] = []
    logger.info("폴백 LLM 생성 사용.")
    _log_trace(step="Composer", purpose="폴백 LLM 생성.", response=state["response"])
    return state
