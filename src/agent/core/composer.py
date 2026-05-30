"""Core Agent — Composer: 서브에이전트 출력을 조합해 최종 응답을 생성.

route별 처리:
- "learn"  : socratic_agent의 <question> 부분만 노출
- "escape" + pending_quiz : 정답 비공개 + 힌트/제출 유도 (LLM 0회)
- "escape" + no quiz : socratic_agent의 <answer> 부분만 노출
- "tools"  : 도구 실행 결과를 LLM으로 합성
- 퀴즈 채점: 사용자가 답안 제출 시 채점 결과 반환

우선순위:
1. 퀴즈 채점 (pending_quiz + 답안 제출)
2. escape + pending_quiz → 정답 비공개 안내 (LLM 없음)
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


def _extract_feedback(text: str) -> str:
    """socratic_agent 출력에서 <feedback> 태그 내용을 추출한다.
    비어있거나 메타 코멘트(피드백 없음 안내 등)면 빈 문자열을 반환한다."""
    match = re.search(r"<feedback>(.*?)</feedback>", text, re.DOTALL)
    if not match:
        return ""
    feedback = match.group(1).strip()
    # LLM이 "피드백 없음" 류의 메타 코멘트를 쓴 경우 무시
    meta_signals = ["피드백을 제공할 수 없", "답변이 없", "첫 번째 질문", "피드백 없음"]
    if any(s in feedback for s in meta_signals):
        return ""
    return feedback


def _extract_question(text: str, has_prior_answer: bool = True) -> str:
    """socratic_agent 출력에서 <feedback> + <question> 태그 내용을 합쳐서 반환한다.
    has_prior_answer=False이면 feedback을 무조건 생략한다.
    태그가 없으면 <answer> 이후 텍스트만, 그것도 없으면 마지막 문장만 반환한다."""
    feedback = _extract_feedback(text) if has_prior_answer else ""
    q_match = re.search(r"<question>(.*?)</question>", text, re.DOTALL)
    if q_match:
        question = q_match.group(1).strip()
        return (f"{feedback}\n\n{question}" if feedback else question).strip()
    # <answer> 태그가 있으면 그 이후 텍스트에서 질문 문장만 추출
    after_answer = re.sub(r"<answer>.*?</answer>", "", text, flags=re.DOTALL).strip()
    if after_answer:
        # LLM이 태그를 그대로 남긴 경우 사용자 응답에 노출되지 않도록 제거
        cleaned = re.sub(r"</?(answer|feedback|question)>", "", after_answer, flags=re.IGNORECASE)
        return cleaned.strip()
    # 최후 fallback: 마지막 문장만 반환
    sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+", text) if s.strip()]
    return sentences[-1] if sentences else text


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

    # ── 2. escape + 퀴즈 중 → 정답 비공개 + 힌트/제출 유도 (LLM 0회) ──
    # 퀴즈 학습 효과를 위해 진행 중에는 정답을 즉시 공개하지 않는다.
    if route == "escape" and pending_quiz:
        state["response"] = (
            "퀴즈 진행 중에는 정답을 바로 공개하지 않아요.\n"
            "'채점하기'로 제출하면 문항별 피드백을 드릴게요.\n"
            "원하면 '힌트 줘'라고 말해주면 정답 없이 풀이 힌트를 줄게요."
        )
        # 퀴즈 상태 유지: 사용자가 이어서 답안을 제출할 수 있어야 함
        state["tool_results"] = []
        logger.info("escape route + pending_quiz → 정답 비공개 안내 응답.")
        _log_trace(step="Composer", purpose="퀴즈 escape 정답 비공개 응답 (LLM 없음).", decision="pending_quiz 유지 + 안내 response.")
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

    # ── 5. learn route → <question> 또는 <answer> 추출 ──────────
    # force_explain=True(반문 한도 초과)이면 <answer>를, 아니면 <question>을 노출한다.
    if tutor_response:
        from src.agent.helpers import _is_summary_request

        is_summary = _is_summary_request(last_msg) or _is_summary_request(state.get("rewritten_query", ""))

        # 현재 사용자 메시지가 '답변'인지 '새 질문'인지 판별
        # 새 질문이면 feedback을 무조건 숨긴다 (LLM이 엉뚱한 칭찬을 만들어내는 것 방지)
        _question_markers = (
            "뭐야", "뭐예요", "뭔가요", "뭔지", "뭔데", "무엇",
            "알려줘", "알려주세요", "설명해줘", "설명해주세요",
            "어떻게", "어떤", "왜", "언제", "누가", "어디",
            "what", "how", "why", "when", "who", "where",
        )
        _last_msg_lower = last_msg.lower().strip()
        _is_new_question = (
            _last_msg_lower.endswith("?") or
            _last_msg_lower.endswith("？") or
            any(_last_msg_lower.startswith(m) or _last_msg_lower.endswith(m)
                for m in _question_markers)
        )
        has_prior_answer = (
            not _is_new_question
            and len(history) >= 2
            and history[-2]["role"] == "assistant"
        )
        
        if is_summary:
            answer = _extract_answer(tutor_response)
            question = _extract_question(tutor_response, has_prior_answer=False)
            state["response"] = f"{answer}\n\n{question}".strip()
            logger.info("요약 요청 → <answer> + <question> 결합 완료.")
            _log_trace(step="Composer", purpose="요약 요청: 결합 추출.", decision="<answer>+<question> → response.")
        elif state.get("force_explain"):
            extracted = _extract_answer(tutor_response)
            state["response"] = extracted or _extract_question(tutor_response, has_prior_answer=False)
            logger.info("반문 한도 초과 → <answer> 추출 완료.")
            _log_trace(step="Composer", purpose="한도 초과: <answer> 추출.", decision="<answer> → response.")
        else:
            state["response"] = _extract_question(tutor_response, has_prior_answer=has_prior_answer)
            logger.info("learn route → <question> 추출 완료.")
            _log_trace(step="Composer", purpose="learn: <question> 추출.", decision="<question> → response.")

        # Safety net: any leftover structured tags should never be shown to end users.
        state["response"] = re.sub(
            r"</?(answer|feedback|question)>",
            "",
            state.get("response", ""),
            flags=re.IGNORECASE,
        ).strip()
        state["tool_results"] = []
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
